from lib.models import OpenAIModel, MistralModel, GigaChatModel, LMStudioAPIModel
from pydantic import BaseModel, Field
from enum import Enum
from typing import Union
import time

from openai import pydantic_function_tool

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


tools = [pydantic_function_tool(CreatePivotTable)]


# model = MistralModel("open-mistral-nemo")
model = OpenAIModel("gpt-4o-mini")
# model = GigaChatModel('GigaChat-Pro')
model = LMStudioAPIModel('llama-3.2-3b-instruct')

print_header("Test 1")
res, usage, probs, times = model.model_response("Who are you?")
print(res), print(usage), print(times)

time.sleep(5)

print_header("Test 2")
res, usage, probs, times = model.model_response("Who are you?", structure=Person)
print(res), print(usage), print(times)

time.sleep(5)

print_header("Test 3")
res, usage, probs, times = model.model_response(
    "Describe any table you want", structure=TableDescription
)
print(res), print(usage), print(times)

time.sleep(5)

print_header("Test 4")
res, usage, probs, times = model.model_response(
    "Who are you?", structure=Person, need_logprobs=True
)
print(res), print(probs), print(usage), print(times)

time.sleep(5)

print_header("Test 5")
res, usage, probs, times = model.model_response(
    "Create a character for a RPG game.",
    structure=Character,
    need_logprobs=True,
    temperature=0.5,
)
print(res), print(probs), print(usage), print(times)

time.sleep(5)

print_header("Test 6")
res, usage, probs, times = model.model_response(
    "Please create pivot table of your choice", tools=tools
)
print(res), print(probs)

time.sleep(5)

messages = [
    {"role": "system", "content": "Please create pivot table of your choice"},
    res,
    model.tool_feedback("Pivot table was created", res.tool_calls[0].id),
]

print_header("Test 7")
print(messages)
res, usage, probs, times = model.model_response(messages, tools=tools)
print(res), print(probs)
