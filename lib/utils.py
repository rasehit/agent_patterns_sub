from functools import wraps
import json
import re
import uuid
from collections.abc import Iterable
from typing import Any, Dict, List

from gigachat.models import Messages
from pydantic import BaseModel, ValidationError


class ParsedMessage(Messages):
    parsed: str | None = None


def generate_structured_prompt(schema):
    description = schema.get("description", "")
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    prompt = f"{description.strip()}\n"
    prompt += "Please provide a JSON object with the following fields:\n"

    definitions = schema.get("definitions", {}) or schema.get("$defs", {})

    def generate_prompt_for_properties(
        properties, required_fields, definitions, indent=0
    ):
        prompt_lines = []
        for field_name, field_info in properties.items():
            field_prompt = process_field(
                field_name, field_info, required_fields, definitions, indent
            )
            prompt_lines.append(field_prompt)
        return "\n".join(prompt_lines)

    def process_field(field_name, field_info, required_fields, definitions, indent):
        indent_str = "  " * indent

        # Initialize field description
        field_desc = field_info.get("description", "")
        if field_desc:
            field_desc = f": {field_desc}"

        # Resolve $ref if present
        while "$ref" in field_info:
            ref = field_info["$ref"]
            ref_name = ref.split("/")[-1]
            field_info = definitions.get(ref_name, {})
            if not field_info:
                break

        # Handle 'anyOf', 'oneOf', 'allOf'
        if "anyOf" in field_info:
            # Process 'anyOf' types
            any_of_prompts = []
            for idx, option in enumerate(field_info["anyOf"]):
                # Optionally resolve $ref in option
                option_info = option
                while "$ref" in option_info:
                    ref = option_info["$ref"]
                    ref_name = ref.split("/")[-1]
                    option_info = definitions.get(ref_name, {})
                    if not option_info:
                        break

                option_type = option_info.get("type", "unknown")

                if option_type == "null":
                    option_prompt = f"{indent_str}  Option {idx+1}: null"
                else:
                    # Process the option without repeating the field_name
                    option_prompt = process_option(option_info, definitions, indent + 2)
                    option_prompt = f"{indent_str}  Option {idx+1}:\n{option_prompt}"

                any_of_prompts.append(option_prompt)
            field_prompt = f"{indent_str}- **{field_name}** (any of the following types){field_desc}"
            field_prompt += "\n" + "\n".join(any_of_prompts)
        elif "oneOf" in field_info:
            # Similar processing for 'oneOf'
            pass
        elif "enum" in field_info:
            enum_values = field_info["enum"]
            field_prompt = f"{indent_str}- **{field_name}** (enum){field_desc}"
            field_prompt += (
                f"\n{indent_str}  Allowed values: {', '.join(map(str, enum_values))}"
            )
        else:
            field_type = field_info.get("type", "unknown")

            if field_type == "object":
                # Process object properties recursively
                sub_properties = field_info.get("properties", {})
                sub_required = field_info.get("required", [])
                field_prompt = f"{indent_str}- **{field_name}** (object){field_desc}"
                if sub_properties:
                    field_prompt += (
                        f"\n{indent_str}  The object has the following fields:"
                    )
                    sub_prompt = generate_prompt_for_properties(
                        sub_properties, sub_required, definitions, indent + 2
                    )
                    field_prompt += f"\n{sub_prompt}"
            elif field_type == "array":
                items = field_info.get("items", {})
                # Resolve $ref in items if present
                while "$ref" in items:
                    ref = items["$ref"]
                    ref_name = ref.split("/")[-1]
                    items = definitions.get(ref_name, {})
                    if not items:
                        break

                item_type = items.get("type", "unknown")
                if item_type == "object":
                    field_prompt = (
                        f"{indent_str}- **{field_name}** (array of objects){field_desc}"
                    )
                    sub_properties = items.get("properties", {})
                    sub_required = items.get("required", [])
                    if sub_properties:
                        field_prompt += (
                            f"\n{indent_str}  Each object has the following fields:"
                        )
                        sub_prompt = generate_prompt_for_properties(
                            sub_properties, sub_required, definitions, indent + 2
                        )
                        field_prompt += f"\n{sub_prompt}"
                else:
                    field_prompt = f"{indent_str}- **{field_name}** (array of {item_type}){field_desc}"
            else:
                # Basic type
                field_prompt = (
                    f"{indent_str}- **{field_name}** ({field_type}){field_desc}"
                )

        return field_prompt

    def process_option(field_info, definitions, indent):
        indent_str = "  " * indent
        field_type = field_info.get("type", "unknown")
        field_desc = field_info.get("description", "")
        if field_desc:
            field_desc = f": {field_desc}"

        if field_type == "object":
            # Process object properties recursively
            sub_properties = field_info.get("properties", {})
            sub_required = field_info.get("required", [])
            option_prompt = f"{indent_str}(object){field_desc}"
            if sub_properties:
                option_prompt += f"\n{indent_str}  The object has the following fields:"
                sub_prompt = generate_prompt_for_properties(
                    sub_properties, sub_required, definitions, indent + 2
                )
                option_prompt += f"\n{sub_prompt}"
        elif field_type == "array":
            items = field_info.get("items", {})
            # Resolve $ref in items if present
            while "$ref" in items:
                ref = items["$ref"]
                ref_name = ref.split("/")[-1]
                items = definitions.get(ref_name, {})
                if not items:
                    break

            item_type = items.get("type", "unknown")
            if item_type == "object":
                option_prompt = f"{indent_str}(array of objects){field_desc}"
                sub_properties = items.get("properties", {})
                sub_required = items.get("required", [])
                if sub_properties:
                    option_prompt += (
                        f"\n{indent_str}  Each object has the following fields:"
                    )
                    sub_prompt = generate_prompt_for_properties(
                        sub_properties, sub_required, definitions, indent + 2
                    )
                    option_prompt += f"\n{sub_prompt}"
            else:
                option_prompt = f"{indent_str}(array of {item_type}){field_desc}"
        else:
            # Basic type
            option_prompt = f"{indent_str}({field_type}){field_desc}"

        return option_prompt

    prompt += generate_prompt_for_properties(properties, required_fields, definitions)

    prompt += "\nEnsure that the JSON object conforms to this schema. Answer in short JSON object."
    return prompt


def extract_json_part(model_output: str) -> str | None:
    match = re.search(r"\{.*\}", model_output, re.DOTALL)
    if match:
        return match.group(0)
    else:
        raise ValueError("No JSON object found in the model output.")


def parse_and_validate_model_output(
    model_output: str, structure: BaseModel
) -> BaseModel | None:
    try:
        data = json.loads(extract_json_part(model_output))
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        raise ValueError("Invalid JSON format in the model output.")
    try:
        return structure.model_validate(data)
    except ValidationError:
        for structure_field in structure.__fields__.keys():
            field_type = structure.__fields__[structure_field].annotation
            if isinstance(field_type, Iterable):
                str_val = [str(val) for val in data[structure_field]]
            else:
                str_val = str(data[structure_field])
            data[structure_field] = str_val
        return structure(**data)


def handle_structured(completion, structure: BaseModel):
    message = ParsedMessage.validate(completion.choices[0].message.__dict__)

    try:
        parsed_model: BaseModel = parse_and_validate_model_output(
            completion.choices[0].message.content, structure
        )
        message.parsed = parsed_model
    except Exception as e:
        print(f"Failed to parse and validate model output: {e}")
        message.parsed = structure()
    completion.choices[0].message = message
    return completion


def get_tool_from_pydantic_chema(model):
    """Converts pydantic BasicModel to the gigachat api tool format"""
    schema = model.model_json_schema()

    name = schema.get("title", "")
    description = schema.get("description", "")

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Extract definitions (nested models) from $defs
    definitions = schema.get("$defs", {})

    # Function to resolve $ref references recursively
    def resolve_refs(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:

                ref = obj["$ref"]
                ref_name = ref.split("/")[-1]
                ref_schema = definitions.get(ref_name, {})

                if "title" in ref_schema:
                    del ref_schema["title"]

                return resolve_refs(ref_schema)
            else:
                obj_copy = {k: v for k, v in obj.items() if k != "title"}
                return {k: resolve_refs(v) for k, v in obj_copy.items()}
        elif isinstance(obj, list):
            return [resolve_refs(item) for item in obj]
        else:
            return obj

    resolved_properties = {}
    for prop_name, prop_schema in properties.items():
        if "title" in prop_schema:
            del prop_schema["title"]
        resolved_properties[prop_name] = resolve_refs(prop_schema)

    tool_schema = {
        "name": name,
        "description": description.strip(),
        "parameters": {
            "type": "object",
            "properties": resolved_properties,
            "required": required,
        },
        "return_parameters": {
            "description": name,
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "result of the function",
                }
            },
            "required": ["response"],
        },
    }

    return tool_schema


def remove_trailing_commas(json_string: str) -> str:
    """
    Removes trailing commas from the JSON-like string to make it valid JSON.

    Args:
        json_string (str): The raw JSON-like string.

    Returns:
        str: The cleaned JSON string without trailing commas.
    """
    # Remove trailing commas before closing braces/brackets
    json_string = re.sub(r",\s*(\}|])", r"\1", json_string)
    return json_string


def generate_func_state_id(txt: str):
    namespace = uuid.NAMESPACE_DNS
    return str(uuid.uuid5(namespace, txt))


def execute_react_tool(
    thoughts: str,
    tool_name: str,
    args: Dict[str, Any],
    available_tools: Dict[str, Any],
) -> List[Any]:
    react_prompt_template = """
    Thought: {thoughts}
    Action: {function_name}({arguments})
    Observation: {observation}
    """

    if tool_name in available_tools:
        try:
            tool = available_tools[tool_name]
            tool_out = tool(**args)
        except Exception as e:
            tool_out = f"Error executing tool {tool_name}: {e} Check the tool names or arguments and try again."
    else:
        tool_out = f"Tool {tool_name} not found. Try again with a valid tool name."

    result_prompt = react_prompt_template.format(
        thoughts=thoughts,
        function_name=tool_name,
        arguments=args,
        observation=tool_out,
    )
    return result_prompt

def _immutable_dialog(dialog):
    """
    Convert dialog list of dicts to a tuple of tuples.
    """
    return tuple(
        tuple(sorted(item.items())) for item in dialog
    )
    
def cache_model_response(func):
    cache = {}
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert dialog to a hashable type
        self_instance = args[0]
        dialog = args[1]
        
        dialog_hashable = _immutable_dialog(dialog)
        key = dialog_hashable

        if key in cache:
            return cache[key]
        
        result = func(*args, **kwargs)
        cache[key] = result
        return result
    return wrapper