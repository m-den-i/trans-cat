from typing import Sequence

import pandas as pd

from lib import transcat
from lib.models import PredictionResult

from .analyzers import Analyzer, RegexDetector, DBCheckDetector
from .redis import RedisMessage
from .db import engine, db_table
from .constants import CATEGORY_COLUMN, FEATURE_VEC
from .response_models import MessageResponse, MessageResponseExisting, BaseMessageResponse
from .storage import SQLCategoriesUpdateStorage, SQLTrainStorage, SQLUpdateStorage


try:
    from application.addons import analyzer_regex
    REGEX_MAP = analyzer_regex.REGEX_MAP
except ImportError:
    analyzer_regex = None
    REGEX_MAP = {}


analyzer = Analyzer(
    detectors=(
        DBCheckDetector(engine, db_table),
        RegexDetector(regex_map=REGEX_MAP),
    ),
)


async def predict_and_save(storage: SQLTrainStorage) -> PredictionResult:
    feature_vec = FEATURE_VEC
    predict = transcat.train_model_predict(storage, feature_vec, CATEGORY_COLUMN)
    update_storage = SQLUpdateStorage(
        train_model=storage.train_model,
        predict_model=storage.df,
        engine=engine,
        table=db_table.name,
    )
    transcat.save_model_predict(update_storage, predict.lable_predict, CATEGORY_COLUMN)
    return predict


async def update_category(index: str, category: str):
    update_storage = SQLCategoriesUpdateStorage(index, engine, db_table.name)
    transcat.save_model_predict(update_storage, [category], CATEGORY_COLUMN)


def get_chosen_category(index) -> tuple[PredictionResult, pd.DataFrame]:
    loader = SQLCategoriesUpdateStorage(index, engine, db_table.name)
    tm_model = loader.load_train_model()
    data = tm_model.loc[tm_model["id"] == index]
    predict = PredictionResult(lable_predict=data[CATEGORY_COLUMN].values, lable_probs=[0], lable_predict_single=[])
    return predict, tm_model.loc[tm_model["id"] == index]


async def check_labels(msgs: Sequence[RedisMessage]):
    found, missing = await analyzer.analyze(msgs)
    return found, missing


async def assign_label_and_get_response(msgs: Sequence[RedisMessage]):
    found, missing = await check_labels(msgs)
    messages: list[BaseMessageResponse] = []
    if missing:
        storage = SQLTrainStorage(missing, engine, db_table.name)
        predict = await predict_and_save(storage)
        messages.append(MessageResponse.from_result(predict, storage.df))
    if found:
        messages.append(MessageResponseExisting.from_result(found))
    return messages
