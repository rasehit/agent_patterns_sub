from openai import OpenAI, pydantic_function_tool
import math
import json

class OpenAIModel:
    def __init__(self, name):
        self.name = name
        self.get_tool_from_pydantic = pydantic_function_tool
    
    def single_query_response(
        self, 
        dialog, 
        tools=None, 
        return_num=1,
        need_logprobs=False,
        structure=None,
        temperature=0.0
    ):
        if isinstance(dialog, str):
            dialog = [{"role": "system", "content": dialog}]
        elif isinstance(dialog, list):
            if isinstance(dialog[0], str):
                dialog = [{"role": "system", "content": dialog[0]}] + \
                    [{"role": "user", "content": message} for message in dialog[1:]]
        else:
            return
        if tools is None:
            completion = self._signle_response(dialog, return_num, structure, temperature)
            if return_num == 1:
                result = completion.choices[0].message
            else:
                result = [choice.message for choice in completion.choices]
        else:
            completion = self._signle_response_tools(dialog, tools, temperature)
            result = completion.choices[0].message
        if need_logprobs and tools is None:
            if return_num == 1:
                probs = [(x.token, math.exp(x.logprob)) for x in completion.choices[0].logprobs.content]
            else:
                probs = [[(x.token, math.exp(x.logprob)) for x in choice.logprobs.content] for choice in completion.choices]
            return (result, completion.usage, probs)
        return (result, completion.usage)
    
    
    def _signle_response(self, messages, return_num, structure, temperature):
        if structure is None:
            completion = OpenAI().chat.completions.create(
                model=self.name,
                messages=messages,
                temperature=temperature,
                n=return_num,
                logprobs=True
            )
        else:
            completion = OpenAI().beta.chat.completions.parse(
                model=self.name,
                messages=messages,
                response_format=structure,
                temperature=temperature,
                n=return_num,
                logprobs=True
            )
        return completion
    
    
    def _signle_response_tools(self, messages, tools, temperature):
        completion = OpenAI().chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=temperature,
            tools=tools,
        )
        return completion       
    
    def tool_feedback(self, result, call_id):
        return {
            "role": "tool",
            "content": result,
            "tool_call_id": call_id
        }