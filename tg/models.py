import abc
from dataclasses import dataclass
from decimal import Decimal
import enum
from typing import Iterable, Protocol, TypedDict, Union
from pydantic import BaseModel

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from lib.models import PredictionResult

import pandas as pd

from tg.redis import CardData, IncreaseData, RedisMessage, TType, TransferData, ValueData


class BaseMessageCategory(TypedDict):
    text: str
    reply_markup: InlineKeyboardMarkup


@dataclass
class BaseMessageResponse:
    message: str
    indexes: list[str]

    @abc.abstractmethod
    def to_fix_category_response(self) -> BaseMessageCategory:
        ...


@dataclass
class MessageResponse(BaseMessageResponse):
    @classmethod
    def from_result(cls, predict: PredictionResult, result: pd.DataFrame) -> 'MessageResponse':
        lines = []
        indexes = []
        for ind, index in enumerate(result.index):
            data = result.loc[index]
            lines.append(f"[{data['id']}] {data['amount']} {data['description']} <> {predict.lable_predict[ind]} ({round(predict.lable_probs[ind])})")
            indexes.append(data["id"])
        text = "\n".join(lines)
        return cls(text, indexes)

    def to_fix_category_response(self):
        keyboard = [
            [
                InlineKeyboardButton(f"{index}", callback_data=index) for index in self.indexes
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        return {"text": self.message, "reply_markup": reply_markup}


class ExistingResponse(BaseModel):
    key: str
    found_key: str
    description: str
    category: str
    amount: str


@dataclass
class MessageResponseExisting(BaseMessageResponse):
    found_indexes: list[str]

    @classmethod
    def from_result(cls, result: list[ExistingResponse]) -> 'MessageResponseExisting':
        lines = []
        indexes = []
        found_indexes = []
        for item in result:
            lines.append(f"[{item.key}] {item.amount} {item.description} <> Found at [{item.found_key}]: {item.category}")
            indexes.append(item.key)
            found_indexes.append(item.found_key)
        text = "\n".join(lines)
        return cls(text, indexes=indexes, found_indexes=found_indexes)
    

    def to_fix_category_response(self):
        keyboard = [
            [
                InlineKeyboardButton(f"{index}", callback_data=index) for index in self.found_indexes
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        return {"text": self.message, "reply_markup": reply_markup}



@enum.unique
class EventAction(str, enum.Enum):
    ASSIGNED = "assigned"
    CHANGED = "changed"


@dataclass
class LabelEvent:
    id: str
    label: str
    action: EventAction = EventAction.ASSIGNED


def get_data(fields, data):
    for f in fields:
        yield getattr(data, f)


def get_description_transfer(data: Union[TransferData, IncreaseData]) -> str:
    return f"{data.title}@{data.name}->{data.destination}"


def get_description_card(data: CardData) -> str:
    return data.place


def get_text_description(res_msg: RedisMessage) -> str:
    if isinstance(res_msg.data, CardData):
        return get_description_card(res_msg.data)
    elif isinstance(res_msg.data, (TransferData, IncreaseData)):
        return get_description_transfer(res_msg.data)
    raise Exception(f"unknown type: {type(res_msg.data)}")


def _get_data_from_message(res_msg: RedisMessage, basic_fields: list):
    data = list(get_data(basic_fields, res_msg.data))
    data.append(get_text_description(res_msg))
    return data + [res_msg.data.type]


def get_df_from_response(msgs: Iterable[RedisMessage]):
    results = []
    basic_fields = ["operation_date", "amount"]
    fields = basic_fields + ["description", "type"]
    for _res in msgs:
        results.append([_res.key] + _get_data_from_message(_res, basic_fields))
    df = pd.DataFrame(results, columns=["id"] + fields)
    return df


# def tuples_to_msgs(tuples: Iterable[tuple[bytes, bytes]]):


def get_df_from_response_tuple(msgs: Iterable[tuple[bytes, bytes]]):
    results = []
    basic_fields = ["operation_date", "amount"]
    fields = basic_fields + ["description", "type"]
    for _res in msgs:
        res_id, res_msg = _res
        results.append([res_id.decode("utf-8")] + _get_data_from_message(res_msg, basic_fields))
    df = pd.DataFrame(results, columns=["id"] + fields)
    return df