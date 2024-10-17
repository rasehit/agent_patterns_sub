import json
import math
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Tuple

from gigachat import GigaChat
from gigachat.models import Chat, Function, FunctionCall, Messages
from mistralai import Mistral
from openai import OpenAI, pydantic_function_tool
from pydantic import BaseModel

from .utils import (
    generate_func_state_id,
    generate_structured_prompt,
    get_tool_from_pydantic_chema,
    handle_structured,
    remove_trailing_commas,
)


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
    def get_text_from_message(self, message: Any) -> str: ...

    """
    Returns the text message of the response.
    """

    @abstractmethod
    def get_structure_from_message(self, message: Any) -> BaseModel: ...

    """
    Returns the structure of the response.
    """

    @abstractmethod
    def get_tool_call_from_message(
        self, message: Any
    ) -> list[tuple[str, dict, str]]: ...

    """
    Returns the tool call of the response.
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
    def get_probs(self, completion: Any) -> list[tuple[Any, float]]: ...

    """
    Returns the tokens probabilities.
    """

    @abstractmethod
    def get_tool_mapping(
        self, tools: list[tuple[BaseModel, Any]]
    ) -> Tuple[list[Any], dict[str, dict]]: ...

    """
    Returns the tools mapping.
    """

    @abstractmethod
    def get_tool_from_pydantic(self, tool: BaseModel) -> Any: ...

    """
    Returns the tools mapping.
    """


class OpenAILikeAPIModel(APIModel):
    def __init__(self, name):
        self.name = name

    def get_text_from_message(self, message: Any) -> str:
        return message.content

    def get_structure_from_message(self, message: Any) -> BaseModel:
        return message.parsed

    def get_usage(self, completion: Any) -> dict:
        return completion.usage

    def get_tool_mapping(
        self, tools: list[tuple[BaseModel, Any]]
    ) -> Tuple[list[Any], dict[str, dict]]:
        llm_tools = [self.get_tool_from_pydantic(tool) for tool, _ in tools]
        call_tools = {
            self.get_tool_from_pydantic(tool)["function"]["name"]: call
            for tool, call in tools
        }
        return llm_tools, call_tools


class OpenAIModel(OpenAILikeAPIModel):
    def __init__(self, name):
        super().__init__(name)

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

    def get_tool_call_from_message(self, message: Any) -> list[tuple[str, dict, str]]:
        tool_calls = message.tool_calls
        if tool_calls is None:
            return []
        return [
            (call.function.name, json.loads(call.function.arguments), call.id)
            for call in tool_calls
        ]

    def tool_feedback(self, result, call_id):
        return {"role": "tool", "content": result, "tool_call_id": call_id}

    def get_probs(self, completion: Any) -> list[tuple[Any, float]]:
        return [
            (x.token, math.exp(x.logprob))
            for x in completion.choices[0].logprobs.content
        ]

    def get_tool_from_pydantic(self, tool: BaseModel) -> Any:
        return pydantic_function_tool(tool)


class MistralModel(OpenAILikeAPIModel):
    def __init__(self, name: str, max_tokens=2048):
        super().__init__(name)
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
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

    def get_tool_call_from_message(self, message: Any) -> list[tuple[str, dict, str]]:
        tool_calls = message.tool_calls
        if tool_calls is None:
            return []
        return [
            (call.function.name, json.loads(call.function.arguments), call.id)
            for call in tool_calls
        ]

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

    def get_tool_from_pydantic(self, tool: BaseModel) -> Any:
        return pydantic_function_tool(tool)


class GigaChatModel(OpenAILikeAPIModel):
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
            temperature=temperature + 0.01,
            max_tokens=self.max_tokens,
        )
        completion = self.giga.chat(chat)
        if structure is not None:
            completion = handle_structured(completion, structure)
        return completion

    def signle_response_tools(self, messages, tools, temperature):
        gigachat_messages = [Messages.validate(message) for message in messages]
        chat = Chat(
            messages=gigachat_messages,
            temperature=temperature + 0.01,
            functions=tools,
            function_call="auto",
            max_tokens=self.max_tokens,
        )
        completion = self.giga.chat(chat)
        msg = completion.choices[0].message
        if self.get_tool_call_from_message(msg) is None:
            completion.choices[0].message = self.parse_tool_call(msg, tools)
        return completion

    def get_structured_query(self, structure, message: dict) -> dict:
        content = message.content
        structure_prompt = generate_structured_prompt(structure.model_json_schema())
        content += f"\n{structure_prompt}"
        schema = message.__dict__
        schema["content"] = content
        return type(message)(**schema)

    def get_tool_call_from_message(self, message: Any) -> list[tuple[str, dict, str]]:
        tool_call = message.function_call
        if tool_call is None:
            return []
        return [(tool_call.name, tool_call.arguments, message.functions_state_id)]

    def parse_tool_call(self, message: Any, tools_schema: List[dict]) -> Messages:

        text_content = message.content
        try:
            cleaned_str = remove_trailing_commas(text_content)
            react_output = json.loads(cleaned_str)
        except json.JSONDecodeError:
            react_output = {
                "thoughts": cleaned_str,
                "function_call": None,
                "args": None,
            }

        thought = react_output.get("thoughts")
        function_name = react_output.get("function_call", "")
        args = react_output.get("args")
        if function_name in tools_schema:
            func_call = FunctionCall(
                name=function_name,
                arguments=args,
            )
            message = Messages(
                role="assistant",
                content=thought,
                function_call=func_call,
                functions_state_id=generate_func_state_id(message.content),
            )

        return message

    def get_tool_from_pydantic(self, tool: BaseModel) -> Function:
        return Function.validate(get_tool_from_pydantic_chema(tool))

    def get_tool_mapping(
        self, tools: list[tuple[BaseModel, Any]]
    ) -> Tuple[list[Any], dict[str, dict]]:
        llm_tools = [self.get_tool_from_pydantic(tool) for tool, _ in tools]
        call_tools = {
            self.get_tool_from_pydantic(tool).name: call for tool, call in tools
        }
        return llm_tools, call_tools

    def get_probs(self, completion: Any):
        return []

    def tool_feedback(self, result, call_id):
        return {"role": "function", "content": result, "functions_state_id": call_id}


class OpenAPIModel(OpenAILikeAPIModel):
    def __init__(self, name, base_url="http://localhost:1234/v1", api_key="lm-studio"):
        super().__init__(name)
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.get_tool_from_pydantic = pydantic_function_tool

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

    def get_tool_call_from_message(self, message: Any) -> list[tuple[str, dict, str]]:
        tool_calls = message.tool_calls
        if tool_calls is None:
            return []
        return [
            (call.function.name, json.loads(call.function.arguments), call.id)
            for call in tool_calls
        ]

    def tool_feedback(self, result, call_id):
        return {"role": "tool", "content": result, "tool_call_id": call_id}

    def get_probs(self, completion: Any) -> list[tuple[Any, float]]:
        return [
            (x.token, math.exp(x.logprob))
            for x in completion.choices[0].logprobs.content
        ]

    def get_tool_from_pydantic(self, tool: BaseModel) -> Any:
        return pydantic_function_tool(tool)
