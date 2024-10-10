import os
import time
from enum import Enum
from typing import Union

import pytest
from openai import pydantic_function_tool
from pydantic import BaseModel, Field

from lib.models import GigaChatModel, MistralModel, OpenAIModel


class Person(BaseModel):
    name: str


class TableType(str, Enum):
    dictionary = "dictionary"
    relational = "relational"
    empty = "empty"


class RelationValue(BaseModel):
    name: str
    description: str
    type: str


class TableDescription(BaseModel):
    table_type: TableType
    content: Union[None, list[RelationValue]]
    table_description: str


class Character(BaseModel):
    name: str
    health_points: int
    game_class: str
    weapons: list[str]
    backstory: str
    favorite_food: str


class PivotPosition(BaseModel):
    column_index: int
    aggregation_function: str


class CreatePivotTable(BaseModel):
    """
    Call this function to create a pivot table with data on the sheet.
    """

    source_range: str = Field(
        description="The name of the source range of pivot table in A1 notation. Defines where data will be read from."
    )
    target_cell: str = Field(
        description="ID of the target cell in A1 notation. Pivot table will be written into this cell."
    )
    rows: list[int] = Field(
        description="""
Column indices in the source range that will be rows in the pivot table. Numbering starts from 0. Should be list of non-negative integers.
For example: [0, 2] means that columns with indices 1 and 3 will be rows in the pivot table.
List can be empty, then there will be no rows in pivot table.
        """
    )
    columns: list[int] = Field(
        description="""
Column indices in the source range that will be columns in the pivot table. Numbering starts from 0. Should be list of non-negative integers.
For example: [1, 3] means that columns with indices 1 and 3 will be columns in the pivot table.
List can be empty, then there will be no columns in pivot table.
        """
    )
    values: list[PivotPosition] = Field(
        description="""
Column indices in the source range that will be data values the pivot table. Numbering starts from 0. 
Should be list of tuples: the first tuple element is column index, the second tuple element is aggregation function.
Aggregation function can be one of these: "SUM", "COUNTA", "AVERAGE", "MAX", "MIN"
For example: [(4, "SUM")] means that columns with index 4 will summed up in the pivot table.
List can be empty, then no values will be written into the pivot table.
        """
    )


def CreatePivotTableRun(source_range, target_cell, rows, columns, values):
    return f"Created pivot table from data in {source_range} with rows {rows}, columns {columns} and values {values}\n"


def print_header(header):
    print(f"\n{header:-^80}\n")


@pytest.fixture
def gigachat_model():
    return GigaChatModel("GigaChat-Pro")


@pytest.fixture
def mistral_model():
    return MistralModel("open-mistral-nemo")


@pytest.fixture
def openai_model():
    return OpenAIModel("gpt-4o-mini")


def assert_single_text_response(
    model, text, structure=None, need_logprobs=False, temperature=0.7, tools=None
):
    message, usage, probs, times = model.model_response(
        text,
        structure=structure,
        need_logprobs=need_logprobs,
        temperature=temperature,
        tools=tools,
    )

    print(f"\nmessage: {message}")

    assert usage.prompt_tokens and usage.completion_tokens and usage.total_tokens
    assert isinstance(times, tuple) or times is None

    if structure is None and tools is None:
        text_content = model.get_text_from_message(message)
        assert isinstance(text_content, str)

    if need_logprobs:
        assert isinstance(probs, list)
        assert all(isinstance(prob, tuple) for prob in probs)

    if structure is not None and tools is None:
        struct = model.get_structure_from_message(message)
        assert isinstance(
            struct, structure
        ), f"Expected {structure}, got {type(struct)}"
        assert structure(**struct.__dict__) is not None

    if tools is not None:
        tool_calls = model.get_tool_call_from_message(message)
        assert tool_calls is not None, "Expected tool call, got None"
        print(f"tool_calls: {tool_calls}")

        if isinstance(tool_calls, list):
            tool_calls = tool_calls[0]

        call_id = (
            tool_calls.id if hasattr(tool_calls, "id") else message.functions_state_id
        )
        messages = [
            {"role": "system", "content": "Please create pivot table of your choice"},
            message,
            model.tool_feedback("Pivot table was created", call_id),
        ]

        time.sleep(5)
        message, usage, probs, times = model.model_response(messages, tools=tools)

        tool_calls = model.get_tool_call_from_message(message)
        print(f"message: {message}")
        print(f"tool_calls: {tool_calls}")
        assert tool_calls is None

    # time.sleep(5)


def test_gigachat(gigachat_model):
    assert_single_text_response(gigachat_model, "Who are you?")
    assert_single_text_response(gigachat_model, "Who are you?", structure=Person)
    assert_single_text_response(
        gigachat_model, "Describe any table you want", structure=TableDescription
    )
    assert_single_text_response(
        gigachat_model,
        "Create a character for a RPG game.",
        structure=Character,
        temperature=0.5,
    )

    tools = [gigachat_model.get_tool_from_pydantic(CreatePivotTable)]
    assert_single_text_response(
        gigachat_model, "Please create pivot table of your choice", tools=tools
    )


def test_mistral(mistral_model):
    assert_single_text_response(mistral_model, "Who are you?")
    assert_single_text_response(mistral_model, "Who are you?", structure=Person)
    assert_single_text_response(
        mistral_model, "Who are you?", structure=Person, need_logprobs=True
    )
    assert_single_text_response(
        mistral_model, "Describe any table you want", structure=TableDescription
    )
    assert_single_text_response(
        mistral_model,
        "Create a character for a RPG game.",
        structure=Character,
        temperature=0.5,
    )

    tools = [mistral_model.get_tool_from_pydantic(CreatePivotTable)]
    assert_single_text_response(
        mistral_model, "Please create pivot table of your choice", tools=tools
    )


def test_openai(openai_model):
    assert_single_text_response(openai_model, "Who are you?")
    assert_single_text_response(openai_model, "Who are you?", structure=Person)
    assert_single_text_response(
        openai_model, "Who are you?", structure=Person, need_logprobs=True
    )
    assert_single_text_response(
        openai_model, "Describe any table you want", structure=TableDescription
    )
    assert_single_text_response(
        openai_model,
        "Create a character for a RPG game.",
        structure=Character,
        temperature=0.5,
    )

    tools = [openai_model.get_tool_from_pydantic(CreatePivotTable)]
    assert_single_text_response(
        openai_model, "Please create pivot table of your choice", tools=tools
    )
