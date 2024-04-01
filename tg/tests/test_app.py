import pytest
from unittest import mock
from lib import transcat

from application.app import update_category, predict_and_save
from application.constants import CATEGORY_COLUMN
from application.storage import SQLTrainStorage
from tg.models import MessageResponse


pytestmark = pytest.mark.asyncio


async def test_msg_conversion(storage: SQLTrainStorage):
    predict = transcat.train_model_predict(storage, ["amount", "description"], CATEGORY_COLUMN)
    assert [round(p) for p in predict.lable_probs] == [58, 89, 49]
    assert [p for p in predict.lable_predict] == ["Zabka Nano", "Carrefour", "Leroy Merlin"]


async def test_update_and_save(storage: SQLTrainStorage):
    with mock.patch("tg.application.db_table.name", storage.table):
        predict = await predict_and_save(storage)
    df = storage._load_train_model()
    got = df.iloc[-3:].to_dict(orient="list")
    assert got["id"] == list(storage.df["id"][-3:])
    assert got[CATEGORY_COLUMN] == predict.lable_predict


async def test_format_message(storage: SQLTrainStorage):
    predict = transcat.train_model_predict(storage, ["amount", "description"], CATEGORY_COLUMN)
    msg = MessageResponse.from_result(predict, storage.df)
    expected = MessageResponse(
        message="[466-0] -6,98 PLN Zabka NANO Warszawa <> Zabka Nano (58)\n[467-0] -245,23 PLN CARREFOUR HIPERMARKET WARSZAWA <> Carrefour (89)\n[468-0] -26,97 PLN Leroy Merlin Warszawa A Warszawa <> Leroy Merlin (49)",
        indexes=["466-0", "467-0", "468-0"]
    )
    assert msg == expected


async def test_update_category(storage: SQLTrainStorage):
    with mock.patch("tg.application.db_table.name", storage.table):
        predict = await predict_and_save(storage)
        await update_category("467-0", "Zabka")
    df = storage._load_train_model()
    got = df.iloc[-1:].to_dict(orient="list")
    assert got["id"] == ["467-0"]
    assert got[CATEGORY_COLUMN] == ["Zabka"]
