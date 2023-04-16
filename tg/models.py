from dataclasses import dataclass
import enum
import json
from typing import Iterable

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from lib.models import PredictionResult

import pandas as pd


@dataclass
class MessageResponse:
    message: str
    indexes: list[str]

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
        yield data[f]


def get_description_transfer(data) -> str:
    return f"{data['title']}@{data['name']}->{data['destination']}"


def get_description_card(data) -> str:
    return data["place"]


def _get_data_from_message(res_msg, basic_fields):
    res_type, res_data = res_msg[b'type'], json.loads(res_msg[b'data'])
    data = list(get_data(basic_fields, res_data))

    if res_type in (b'SETTLEMENT', b'BLOCKADE'):
        data.append(get_description_card(res_data))
    elif res_type in (b'INCREASE', b'TRANSFER'):
        data.append(get_description_transfer(res_data))

    return data + [res_type.decode("utf-8")]


def get_df_from_response(msgs: Iterable[tuple[bytes, bytes]]):
    results = []
    basic_fields = ["operation_date", "amount"]
    fields = basic_fields + ["description", "type"]
    for _res in msgs:
        res_id, res_msg = _res
        results.append([res_id.decode("utf-8")] + _get_data_from_message(res_msg, basic_fields))
    df = pd.DataFrame(results, columns=["id"] + fields)
    return df
