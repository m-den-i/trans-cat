from dataclasses import dataclass

import pandas as pd

from lib.models import PredictionResult

from application.models import ExistingResponse


@dataclass
class BaseMessageResponse:
    message: str
    indexes: list[str]
    categories: list[str]


@dataclass
class MessageResponse(BaseMessageResponse):
    @classmethod
    def from_result(cls, predict: PredictionResult, result: pd.DataFrame) -> 'MessageResponse':
        lines = []
        indexes = []
        categories = []
        for ind, index in enumerate(result.index):
            data = result.loc[index]
            new_cat = predict.lable_predict[ind]
            lines.append(f"[{data['id']}] {data['amount']} {data['description']} <> {new_cat} ({round(predict.lable_probs[ind])})")
            indexes.append(data["id"])
            categories.append(new_cat)
        text = "\n".join(lines)
        return cls(text, indexes, categories)


@dataclass
class MessageResponseExisting(BaseMessageResponse):
    found_indexes: list[str]

    @classmethod
    def from_result(cls, result: list[ExistingResponse]) -> 'MessageResponseExisting':
        lines = []
        indexes = []
        found_indexes = []
        categories = []
        for item in result:
            lines.append(f"[{item.key}] {item.amount} {item.description} <> Found at [{item.found_key}]: {item.category}")
            indexes.append(item.key)
            found_indexes.append(item.found_key)
            categories.append(item.category)
        text = "\n".join(lines)
        return cls(text, indexes=indexes, found_indexes=found_indexes, categories=categories)
