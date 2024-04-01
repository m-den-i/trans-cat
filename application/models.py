import enum
from dataclasses import dataclass
from typing import Iterable, Union

import pandas as pd
from pydantic import BaseModel

from .redis import RedisMessage, CardData, TransferData, IncreaseData


class ExistingResponse(BaseModel):
    key: str
    found_key: str
    description: str
    category: str
    amount: str


@enum.unique
class EventAction(str, enum.Enum):
    ASSIGNED = "assigned"
    CHANGED = "changed"


@dataclass
class LabelEvent:
    id: str
    label: str
    action: EventAction = EventAction.ASSIGNED


def get_df_from_response(msgs: Iterable[RedisMessage]):
    results = []
    basic_fields = ["operation_date", "amount"]
    fields = basic_fields + ["description", "type"]
    for _res in msgs:
        results.append([_res.key] + get_data_from_message(_res, basic_fields))
    df = pd.DataFrame(results, columns=["id"] + fields)
    return df


def get_data_from_message(res_msg: RedisMessage, basic_fields: list):
    data = list(get_data(basic_fields, res_msg.data))
    data.append(get_text_description(res_msg))
    return data + [res_msg.data.type]


def get_data(fields, data):
    for f in fields:
        yield getattr(data, f)


def get_text_description(res_msg: RedisMessage) -> str:
    if isinstance(res_msg.data, CardData):
        return get_description_card(res_msg.data)
    elif isinstance(res_msg.data, (TransferData, IncreaseData)):
        return get_description_transfer(res_msg.data)
    raise Exception(f"unknown type: {type(res_msg.data)}")


def get_description_transfer(data: Union[TransferData, IncreaseData]) -> str:
    return f"{data.title}@{data.name}->{data.destination}"


def get_description_card(data: CardData) -> str:
    return data.place


def get_df_from_response_tuple(msgs: Iterable[tuple[bytes, bytes]]):
    results = []
    basic_fields = ["operation_date", "amount"]
    fields = basic_fields + ["description", "type"]
    for _res in msgs:
        res_id, res_msg = _res
        results.append([res_id.decode("utf-8")] + get_data_from_message(res_msg, basic_fields))
    df = pd.DataFrame(results, columns=["id"] + fields)
    return df
