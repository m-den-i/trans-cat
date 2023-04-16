from dataclasses import asdict
import os
import pandas as pd
import sqlalchemy as sa

from lib import transcat
from lib.models import PredictionResult
from tg.constants import CATEGORY_COLUMN, FEATURE_VEC
from tg.models import EventAction, LabelEvent
from tg.redis import RedisWriter
from tg.storage import CategoriesUpdateStorage, CustomTrainStorage, CustomUpdateStorage


engine = sa.create_engine(os.environ["DB_URL"])
db_table = os.environ["DB_TABLE"]


async def predict_and_save(storage: CustomTrainStorage, redis_writer: RedisWriter) -> PredictionResult:
    feature_vec = FEATURE_VEC
    predict = transcat.train_model_predict(storage, feature_vec, CATEGORY_COLUMN)
    update_storage = CustomUpdateStorage(
        train_model=storage.train_model,
        predict_model=storage.df,
        engine=engine,
        table=db_table,
    )
    transcat.save_model_predict(update_storage, predict.lable_predict, CATEGORY_COLUMN)
    for ind in storage.df.index:
        data = storage.df.loc[ind]
        await redis_writer.write_to_stream(asdict(LabelEvent(id=data["id"], label=data[CATEGORY_COLUMN])))
    return predict


async def update_category(index: str, category: str, redis_writer: RedisWriter):
    update_storage = CategoriesUpdateStorage(index, engine, db_table)
    transcat.save_model_predict(update_storage, [category], CATEGORY_COLUMN)
    await redis_writer.write_to_stream(asdict(LabelEvent(id=index, label=category, action=EventAction.CHANGED)))


def get_chosen_category(index) -> tuple[PredictionResult, pd.DataFrame]:
    loader = CategoriesUpdateStorage(index, engine, db_table)
    tm_model = loader.load_train_model()
    data = tm_model.loc[tm_model["id"] == index]
    predict = PredictionResult(lable_predict=data[CATEGORY_COLUMN].values, lable_probs=[0], lable_predict_single=[])
    return predict, tm_model.loc[tm_model["id"] == index]
