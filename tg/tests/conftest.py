import asyncio
from datetime import datetime
import os
import pickle
from unittest import mock
import pandas as pd
import pytest
import pytest_asyncio
import sqlalchemy as sa

from redis import Redis
from server.app import get_client, get_service

from settings import UPLOAD_FOLDER

from tg.app import update_category, engine, db_table
from tg.redis import CardData, RedisMessage, RedisReader, RedisWriter, RedisWriterConfig, TType, TransferData, init_redis
from tg.storage import SQLTrainStorage, SQLUpdateStorage


pytestmark = pytest.mark.asyncio


TEST_STREAM = "test_write_stream"
TEST_GROUP = "test_group"


def _get_path(name):
    return os.path.join(UPLOAD_FOLDER, name)


@pytest_asyncio.fixture
async def redis_stream(redis_client: Redis):
    res = await redis_client.xrange(TEST_STREAM, "-", "+", count=2)
    await redis_client.delete(TEST_STREAM)
    await redis_client.xgroup_create(TEST_STREAM, TEST_GROUP, "$", mkstream=True)
    yield redis_client
    await redis_client.delete(TEST_STREAM)


@pytest_asyncio.fixture
async def redis_writer(redis_stream: Redis):
    return RedisWriter(redis_stream, RedisWriterConfig(stream=TEST_STREAM))


@pytest_asyncio.fixture
async def storage(redis: RedisReader):
    db_table_test = f"{db_table.name}_test" 
    test_data = pd.read_csv(_get_path("Marked_test.csv"))
    with engine.connect() as conn:
        conn.execute(sa.text("DROP TABLE IF EXISTS labels_test;"))
        conn.execute(sa.text("CREATE TABLE labels_test AS TABLE labels WITH NO DATA;"))
        conn.commit()
    test_data["operation_date"] = test_data["operation_date"].map(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    upd_storage = SQLUpdateStorage(pd.DataFrame(), pd.DataFrame(), engine, table=db_table_test)
    upd_storage.save_train_model(test_data)
    msgs = await redis.read_messages(index="465", count=3)
    yield SQLTrainStorage(msgs, engine, db_table_test)
    with engine.connect() as conn:
        conn.execute(sa.text("DROP TABLE labels_test"))
        conn.commit()


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


fake_redis = mock.Mock(RedisReader)

async def read_group_pending(*args, **kwargs):
    return await RedisReader.read_group_pending(fake_redis, *args, **kwargs)

fake_redis.read_group_pending = read_group_pending


class FakeContext:
    def __init__(self) -> None:
        self.bot = mock.AsyncMock()
        self.bot_data = {"redis": fake_redis}    


@pytest_asyncio.fixture(scope="session")
async def tg_context():
    fake_ctx = FakeContext()
    fake_ctx.bot_data["redis"].smembers = mock.AsyncMock(return_value=[b"123"])
    return fake_ctx


@pytest_asyncio.fixture(scope="session")
def load_messages():
    with open("x.pickle", "rb") as f:
        return pickle.load(f)


@pytest_asyncio.fixture(scope="function")
async def tg_context_redis(tg_context, load_messages):
    with mock.patch.object(tg_context.bot_data["redis"], "_read_group_pending", side_effect=[load_messages, []]):
        with mock.patch.object(tg_context.bot_data["redis"], "ack", side_effect=[load_messages, []]):
            yield FakeContext()
            pass


@pytest_asyncio.fixture(scope="session")
async def test_msgs():
    return [
        RedisMessage(
            key="1000-0",
            operation_id=1000,
            data=CardData(
                amount="-10,60 PLN",
                operation_date="20-03-2023",
                type=TType.BLOCKADE,
                card="Visa Test 1",
                place="Zabka NANO Warszawa",
            )
        ),
        RedisMessage(
            key="1001-0",
            operation_id=1001,
            data=TransferData(
                type=TType.TRANSFER,
                amount="71,92 PLN",
                operation_date="20-03-2023",
                source="62 1090 1447 0000 0001 4793 2084",
                destination="62 1090 1447 0000 0001 4793 2085",
                name="KRAJOWY INTEGRATOR PŁATNOŚCI SA",
                title="TR-1ABC-8KAM123",
                settlement_date="20-03-2023",
                saldo="100 PLN",
            )
        ),
        RedisMessage(
            key="1002-0",
            operation_id=1002,
            data=CardData(
                amount="-10,60 PLN",
                operation_date="20-03-2023",
                type=TType.SETTLEMENT,
                card="Visa Test 1",
                place="Zabka NANO Warszawa",
            )
        ),
    ]


@pytest_asyncio.fixture(scope="session")
async def service():
    _service = await get_service()
    yield _service
    _service.close()


@pytest_asyncio.fixture(scope="session")
async def client(service):
    _client = await get_client()
    yield _client
    _client.close()
