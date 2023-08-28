from typing import Sequence, Iterable
import pandas as pd
import sqlalchemy as sa

from lib import transcat
from lib.models import PredictionResult
from settings import DB_TABLE, DB_URL
from tg.constants import CATEGORY_COLUMN, FEATURE_VEC
from tg.redis import RedisMessage
from tg.storage import SQLCategoriesUpdateStorage, SQLTrainStorage, SQLUpdateStorage
from tg.models import get_text_description, ExistingResponse


engine = sa.create_engine(DB_URL)
_META_DATA = sa.MetaData()
_META_DATA.reflect(bind=engine)
db_table = _META_DATA.tables[DB_TABLE]


def post_proc_descriptions(descriptions: Iterable[str]) -> tuple[str]:
    result = []
    for desc in descriptions:
        if "BOLT" in desc:
            parts = desc.split(" ")
            first_part, city = "/".join(parts[:-1]), parts[-1]
            result.append(f"{first_part} {city}")
        else:
            result.append(desc)
    return tuple(result)


async def _check_labels(descriptions: Iterable[str]):
    descriptions = tuple(descriptions)
    _id, _description, _category = [getattr(db_table.c, c) for c in ("id", "description", "category")]
    query = sa.select(_id, _description, _category).where(_description.in_(descriptions))
    with engine.connect() as connection:
        result = tuple(connection.execute(query))
    data = {
        res[1]: (res[0], res[2]) for res in result
    }
    found: list[list[int, tuple[str, str]]] = []
    missing: list[int] = []
    for ind, description in enumerate(descriptions):
        if (id_category := data.get(description)) is not None:
            found.append([ind, id_category])
        else:
            missing.append(ind)
    return found, missing


async def check_labels(msgs: Sequence[RedisMessage]):
    found: list[ExistingResponse] = []
    missing: list[RedisMessage] = []
    descriptions = tuple(get_text_description(msg) for msg in msgs)
    found_, missing_ = await _check_labels(descriptions)
    # Fallback
    if missing_:
        descriptions = post_proc_descriptions(descriptions)
        found_, missing_ = await _check_labels(descriptions)
    for ind, id_category in found_:
        msg = msgs[ind]
        id_, category = id_category
        response = ExistingResponse(
            key=msg.key,
            found_key=id_,
            description=descriptions[ind],
            category=category,
            amount=msg.data.amount,
        )
        found.append(response)
    for ind in missing_:
        missing.append(msgs[ind])
    return found, missing


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
