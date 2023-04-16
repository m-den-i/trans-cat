import asyncio
import codecs
import os
import sqlalchemy as sa
from unittest import mock
import pandas as pd
import pytest
import pytest_asyncio
from redis import Redis
from lib import transcat
from tg.app import predict_and_save
from tg.cmd import put_into_stream
from tg.constants import CATEGORY_COLUMN
from tg.models import EventAction, LabelEvent, MessageResponse

from tg.redis import RedisReader, RedisReaderConfig, RedisWriter, RedisWriterConfig, init_redis
from tg.storage import CustomTrainStorage, CustomUpdateStorage
from tg.app import update_category, engine, db_table


pytestmark = pytest.mark.asyncio

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="session")
async def redis_client():
    client = await init_redis()
    assert await client.ping()
    yield client
    await client.close()

@pytest_asyncio.fixture(scope="session")
async def redis(redis_client: Redis):
    return RedisReader(redis_client)


TEST_STREAM = "test_write_stream"
TEST_GROUP = "test_group"

def _get_path(name):
    return os.path.join(os.environ['UPLOAD_FOLDER'], name)

@pytest_asyncio.fixture
async def redis_stream(redis_client: Redis):
    try:
        res = await redis_client.xrange(TEST_STREAM, "-", "+", count=2)
        await redis_client.delete(TEST_STREAM)
    except Exception:
        pass
    await redis_client.xgroup_create(TEST_STREAM, TEST_GROUP, "$", mkstream=True)
    yield redis_client
    await redis_client.delete(TEST_STREAM)

@pytest_asyncio.fixture
async def redis_writer(redis_stream: Redis):
    return RedisWriter(redis_stream, RedisWriterConfig(stream=TEST_STREAM))

@pytest_asyncio.fixture
async def storage(redis: RedisReader):
    db_table_test = f"{db_table}_test" 
    test_data = pd.read_csv(_get_path("Marked_test.csv"))
    with engine.connect() as conn:
        conn.execute(sa.text("CREATE TABLE labels_test AS TABLE labels WITH NO DATA;"))
        conn.commit()
    upd_storage = CustomUpdateStorage(pd.DataFrame(), pd.DataFrame(), engine, table=db_table_test)
    upd_storage.save_train_model(test_data)
    msgs = await redis.read_messages(index="465", count=3)
    yield CustomTrainStorage(msgs, engine, db_table_test)
    with engine.connect() as conn:
        conn.execute(sa.text("DROP TABLE labels_test"))
        conn.commit()
    pass


async def test_msg_conversion(storage: CustomTrainStorage):
    predict = transcat.train_model_predict(storage, ["amount", "description"], CATEGORY_COLUMN)
    assert [round(p) for p in predict.lable_probs] == [58, 89, 49]
    assert [p for p in predict.lable_predict] == ["Zabka Nano", "Carrefour", "Leroy Merlin"]

async def test_update_and_save(storage: CustomTrainStorage, redis_writer: RedisWriter):
    with mock.patch("tg.app.db_table", storage.table):
        predict = await predict_and_save(storage, redis_writer)
    df = storage._load_train_model()
    got = df.iloc[-3:].to_dict(orient="list")
    assert got["id"] == list(storage.df["id"][-3:])
    assert got[CATEGORY_COLUMN] == predict.lable_predict

async def test_format_message(storage: CustomTrainStorage,):
    predict = transcat.train_model_predict(storage, ["amount", "description"], CATEGORY_COLUMN)
    msg = MessageResponse.from_result(predict, storage.df)
    expected = MessageResponse(
        message="[466-0] -6,98 PLN Zabka NANO Warszawa <> Zabka Nano (58)\n[467-0] -245,23 PLN CARREFOUR HIPERMARKET WARSZAWA <> Carrefour (89)\n[468-0] -26,97 PLN Leroy Merlin Warszawa A Warszawa <> Leroy Merlin (49)",
        indexes=["466-0", "467-0", "468-0"]
    )
    assert msg == expected

async def test_update_category(storage: CustomTrainStorage, redis_writer: RedisWriter):
    with mock.patch("tg.app.db_table", storage.table):
        predict = await predict_and_save(storage, redis_writer)
        await update_category("467-0", "Zabka", redis_writer)
    df = storage._load_train_model()
    got = df.iloc[-1:].to_dict(orient="list")
    assert got["id"] == ["467-0"]
    assert got[CATEGORY_COLUMN] == ["Zabka"]

def _dict_from_bytes(data):
    _dict = {}
    for k, v in data.items():
        _dict[k.decode()] = v.decode()
    return _dict

async def test_predict_and_save_sends_event(storage: CustomTrainStorage, redis_writer: RedisWriter):
    with mock.patch("tg.app.db_table", storage.table):
        predict = await predict_and_save(storage, redis_writer)
    reader = RedisReader(redis_writer.client, RedisReaderConfig(stream=TEST_STREAM, group=TEST_GROUP, consumer="Test"))
    res = await reader.read_messages(0, count=5)

    produced = [LabelEvent(**_dict_from_bytes(msg[1])) for msg in res]
    assert produced == [
        LabelEvent('466-0', 'Zabka Nano'),
        LabelEvent('467-0', 'Carrefour'),
        LabelEvent('468-0', 'Leroy Merlin'),
    ]

async def test_update_category_sends_event(storage: CustomTrainStorage, redis_writer: RedisWriter):
    with mock.patch("tg.app.db_table", storage.table):
        predict = await predict_and_save(storage, redis_writer)
        await update_category("467-0", "Zabka", redis_writer)
    reader = RedisReader(redis_writer.client, RedisReaderConfig(stream=TEST_STREAM, group=TEST_GROUP, consumer="Test"))
    res = await reader.read_messages(0, count=5)

    produced = [LabelEvent(**_dict_from_bytes(msg[1])) for msg in res]
    assert produced == [
        LabelEvent('466-0', 'Zabka Nano'),
        LabelEvent('467-0', 'Carrefour'),
        LabelEvent('468-0', 'Leroy Merlin'),
        LabelEvent('467-0', 'Zabka', EventAction.CHANGED)
    ]

async def test_put_into_stream_command(storage: CustomTrainStorage, redis_writer: RedisWriter):
    with mock.patch("tg.cmd.put_into_stream.db_table", storage.table):
        await put_into_stream.put_into_stream.callback(TEST_STREAM, '464-0')
    reader = RedisReader(redis_writer.client, RedisReaderConfig(stream=TEST_STREAM, group=TEST_GROUP, consumer="Test"))
    res = await reader.read_messages(0, count=5)

    produced = [LabelEvent(**_dict_from_bytes(msg[1])) for msg in res]
    assert produced == [
        LabelEvent('464-0', 'Free Now'),
        LabelEvent('465-0', 'Uber Eats'),
    ]
