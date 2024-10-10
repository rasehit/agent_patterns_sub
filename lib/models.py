import os
import math

from abc import ABC, abstractmethod
from typing import Any, List
from pydantic import BaseModel
from datetime import datetime

from openai import OpenAI
from openai import pydantic_function_tool
from mistralai import Mistral
from gigachat import GigaChat
from gigachat.models import (
    Chat,
    Function,
    Messages,
)

from .utils import handle_structured, generate_structured_prompt, get_tool_from_pydantic

class APIModel(ABC):
    def __init__(self, name):
        self.name = name

    def model_response(
        self,
        dialog: str | List[dict],
        tools: List[dict] | None = None,
        need_logprobs: bool = False,
        structure: BaseModel | None = None,
        temperature: float = 0.0,
    ):
        """
        Get model response with the given dialog.

        Args:
            dialog (str | List[dict]): The dialog to query.
            tools (List[dict], optional): Tools to use. Defaults to None.
            need_logprobs (bool, optional): Whether to return logprobs. Defaults to False.
            structure (BaseModel, optional): The structure of the response. Defaults to None.
            temperature (float, optional): The temperature of the response. Defaults to 0.0.
        """
        result, probs = None, None

        if isinstance(dialog, str):
            dialog = [{"role": "system", "content": dialog}]
        elif isinstance(dialog, list):
            if isinstance(dialog[0], str):
                dialog = [{"role": "system", "content": dialog[0]}] + [
                    {"role": "user", "content": message} for message in dialog[1:]
                ]
        else:
            return

        if tools is None:
            start = datetime.utcnow()
            completion = self.signle_response(dialog, structure, temperature)
            end = datetime.utcnow()
            
            result = (
                self.get_structure(completion)
                if structure is not None
                else self.get_text_message(completion)
            )
        else:
            start = datetime.utcnow()
            completion = self.signle_response_tools(dialog, tools, temperature)
            end = datetime.utcnow()
            result = completion.choices[0].message

        usage = self.get_usage(completion)
        created = datetime.utcfromtimestamp(completion.created)
        time_llm = (created - start).total_seconds()
        time_network = (end - created).total_seconds()

        if need_logprobs and tools is None:
            probs = self.get_probs(completion)

        return (result, usage, probs, (time_llm, time_network))

    @abstractmethod
    def get_text_message(self, completion: Any) -> str: ...

    """
    Returns the text message of the response.
    """

    @abstractmethod
    def get_structure(self, completion: Any) -> BaseModel: ...

    """
    Returns the structure of the response.
    """

    @abstractmethod
    def get_usage(self, completion: Any) -> dict: ...

    """
    Returns query token usage.
    """

    @abstractmethod
    def signle_response(self, messages, structure, temperature): ...

    """
    Returns the chat complitions of a single query.
    """

    @abstractmethod
    def signle_response_tools(self, messages, tools, temperature): ...

    """
    Returns the chat complition of a single query with tools.
    """

    @abstractmethod
    def tool_feedback(self, result, call_id): ...

    """
    Return the tool feedback in a proper format.
    """

    @abstractmethod
    def get_probs(self, completion: Any) -> list: ...

    """
    Returns the tokens probabilities.
    """


class OpenAIModel(APIModel):
    def __init__(self, name):
        super().__init__(name)
        self.get_tool_from_pydantic = pydantic_function_tool

    def get_text_message(self, completion: Any) -> str:
        return completion.choices[0].message.content

    def get_structure(self, completion: Any) -> BaseModel:
        return completion.choices[0].message.parsed

    def get_usage(self, completion: Any) -> dict:
        return completion.usage

    def signle_response(self, messages, structure, temperature):
        if structure is None:
            completion = OpenAI().chat.completions.create(
                model=self.name,
                messages=messages,
                temperature=temperature,
                logprobs=True,
            )
        else:
            completion = OpenAI().beta.chat.completions.parse(
                model=self.name,
                messages=messages,
                response_format=structure,
                temperature=temperature,
                logprobs=True,
            )
        return completion

    def signle_response_tools(self, messages, tools, temperature):
        completion = OpenAI().chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=temperature,
            tools=tools,
        )
        return completion

    def tool_feedback(self, result, call_id):
        return {"role": "tool", "content": result, "tool_call_id": call_id}

    def get_probs(self, completion: Any):
        return [
            (x.token, math.exp(x.logprob))
            for x in completion.choices[0].logprobs.content
        ]


class MistralModel(APIModel):
    def __init__(self, name: str, max_tokens=2048):
        super().__init__(name)
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.get_tool_from_pydantic = pydantic_function_tool
        self.max_tokens = max_tokens

    def signle_response(self, messages, structure, temperature):
        if structure is not None:
            last_message = messages[-1]
            last_message = self.get_structured_query(structure, last_message)
            messages[-1] = last_message

            completion = self.client.chat.complete(
                model=self.name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                response_format={
                    "type": "json_object",
                },
            )
            completion = handle_structured(completion, structure)
        else:
            completion = self.client.chat.complete(
                model=self.name,
                messages=messages,
                temperature=temperature,
            )
        return completion

    def signle_response_tools(self, messages, tools, temperature):
        completion = self.client.chat.complete(
            model=self.name,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice="any",
        )
        return completion

    def get_text_message(self, completion: Any) -> str:
        return completion.choices[-1].message.content

    def get_structure(self, completion: Any) -> BaseModel:
        return completion.choices[-1].message.parsed

    def get_usage(self, completion: Any) -> dict:
        return completion.usage

    def get_structured_query(self, structure, message: dict) -> dict:
        content = message["content"]
        structure_prompt = generate_structured_prompt(structure.model_json_schema())
        content += f"\n{structure_prompt}"
        message["content"] = content
        return message

    def get_probs(self, completion: Any):
        return []

    def tool_feedback(self, result, call_id):
        return {"role": "tool", "content": result, "tool_call_id": call_id}


class GigaChatModel(APIModel):
    def __init__(self, name: str, max_tokens=2048):
        super().__init__(name)
        self.giga = GigaChat(model=self.name)
        self.max_tokens = max_tokens

    def signle_response(self, messages, structure, temperature):
        gigachat_messages = [Messages.validate(message) for message in messages]
        if structure is not None:
            last_message = gigachat_messages[-1]
            last_message = self.get_structured_query(structure, last_message)
            gigachat_messages[-1] = last_message
        chat = Chat(
            messages=gigachat_messages,
            temperature=temperature+0.01,
            max_tokens=self.max_tokens
        )
        completion = self.giga.chat(chat)
        if structure is not None:
            completion = handle_structured(completion, structure)
        return completion

    def signle_response_tools(self, messages, tools, temperature):
        gigachat_messages = [Messages.validate(message) for message in messages]
        chat = Chat(
            messages=gigachat_messages,
            temperature=temperature+0.01,
            functions=tools,
            function_call="auto",
            max_tokens=self.max_tokens
        )
        completion = self.giga.chat(chat)
        return completion

    def get_text_message(self, completion: Any) -> str:
        return completion.choices[-1].message.content

    def get_structure(self, completion: Any) -> BaseModel:
        return completion.choices[-1].message.parsed

    def get_usage(self, completion: Any) -> dict:
        return completion.usage

    def get_structured_query(self, structure, message: dict) -> dict:
        content = message.content
        structure_prompt = generate_structured_prompt(structure.model_json_schema())
        content += f"\n{structure_prompt}"
        schema = message.__dict__
        schema["content"] = content
        return type(message)(**schema)

    def get_tool_from_pydantic(self, tool: BaseModel) -> Function:
        return Function.validate(get_tool_from_pydantic(tool))
    
    def get_probs(self, completion: Any):
        return []

    def tool_feedback(self, result, call_id):
        return {"role": "function", "content": result, "functions_state_id": call_id}


class LMStudioAPIModel(APIModel):
    def __init__(self, name):
        super().__init__(name)
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.get_tool_from_pydantic = pydantic_function_tool

    def get_text_message(self, completion: Any) -> str:
        return completion.choices[0].message.content

    def get_structure(self, completion: Any) -> BaseModel:
        return completion.choices[0].message.parsed

    def get_usage(self, completion: Any) -> dict:
        return completion.usage

    def signle_response(self, messages, structure, temperature):
        if structure is None:
            completion = self.client.chat.completions.create(
                model=self.name,
                messages=messages,
                temperature=temperature,
                logprobs=True,
            )
        else:
            completion = self.client.beta.chat.completions.parse(
                model=self.name,
                messages=messages,
                response_format=structure,
                temperature=temperature,
                logprobs=True,
            )
        return completion

    def signle_response_tools(self, messages, tools, temperature):
        completion = self.client.chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=temperature,
            tools=tools,
        )
        return completion

    def tool_feedback(self, result, call_id):
        return {"role": "tool", "content": result, "tool_call_id": call_id}

    def get_probs(self, completion: Any):
        return []