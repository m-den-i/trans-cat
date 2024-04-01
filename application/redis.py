import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pydantic import BaseModel, Field, validator
from typing import Annotated, Literal, Optional, Sequence, Union
import redis.asyncio as redis

from settings import REDIS_READER_SETTINGS, REDIS_URL, REDIS_WRITER_SETTINGS


class TType(StrEnum):
    TRANSFER = "TRANSFER"
    INCREASE = "INCREASE"
    BLOCKADE = "BLOCKADE"
    SETTLEMENT = "SETTLEMENT"


class Value(BaseModel):
    amount: str
    operation_date: datetime
    
    @validator("operation_date", pre=True)
    def parse_operation_date(cls, value):
        if isinstance(value, datetime):
            return value
        return datetime.strptime(
            value,
            "%d-%m-%Y"
        )


class CardData(Value):
    type: Literal[TType.BLOCKADE, TType.SETTLEMENT]
    card: str
    place: str


class TransferData(Value):
    type: Literal[TType.TRANSFER]
    source: str 
    destination: str
    name: str 
    title: str 
    settlement_date: datetime 
    saldo: str

    @validator("settlement_date", pre=True)
    def parse_settlement_date(cls, value):
        if isinstance(value, datetime):
            return value
        elif isinstance(value, str):
            return datetime.strptime(
                value,
                "%d-%m-%Y"
            )
        raise ValueError("Should be str or datetime.")


class IncreaseData(Value):
    type: Literal[TType.INCREASE]
    destination: str
    name: str 
    title: str 
    saldo: str


ValueData = Annotated[Union[CardData, TransferData, IncreaseData], Field(discriminator='type')]


class RedisMessage(BaseModel):
    key: str
    operation_id: int
    data: ValueData


MsgData = tuple[bytes, dict[bytes, bytes]]


def message_from_msg_data(msg_item: MsgData) -> RedisMessage:
    dt_type = TType(msg_item[1][b'type'].decode())
    dt = json.loads(msg_item[1][b'data'])
    return RedisMessage(
        key=msg_item[0],
        operation_id=msg_item[1][b'operation_id'],
        data={"type": dt_type, **dt}
    )


def messages_from_read_resp(resp) -> Sequence[RedisMessage]:
    result = []
    if len(resp):
        _stream_name, msg_data = resp[0]
        for msg_item in msg_data:
            result.append(message_from_msg_data(msg_item))
    return result


async def init_redis() -> redis.Redis:
    return await redis.from_url(REDIS_URL)


@dataclass
class RedisWriterConfig:
    stream: str


@dataclass
class RedisReaderConfig:
    stream: str
    group: str
    consumer: str


class RedisReader:
    def __init__(self, client: redis.Redis, config: Optional[RedisReaderConfig] = None):
        self._client = client
        if config is None:
            config = RedisReaderConfig(**json.loads(REDIS_READER_SETTINGS))
        self.stream = config.stream
        self.group = config.group
        self.consumer = config.consumer
    
    async def smembers(self, value: str):
        return await self._client.smembers(value)

    async def close(self):
        await self._client.close()

    async def read_messages(self, index: str, count: int = 1):
        resp = await self._client.xread({self.stream: index}, count=count)
        return messages_from_read_resp(resp)

    async def read_group_messages(self, index: str = ">", count: int = 1):
        resp = await self._client.xreadgroup(self.group, self.consumer, streams={self.stream: index}, count=count)
        return messages_from_read_resp(resp)

    async def _read_group_pending(self, count: int):
        msgs = []
        resp = await self._client.xpending_range(self.stream, self.group, "-", "+", count, self.consumer)
        for msg in resp:
            res_ = await self._client.xrange(self.stream, msg["message_id"], "+", count=count)
            msgs.append(res_[0]) 
        return msgs

    async def read_group_pending(self, count: int = 1) -> Sequence[RedisMessage]:
        msgs = await self._read_group_pending(count)
        return [message_from_msg_data(msg_item) for msg_item in msgs]
    
    async def ack(self, msg_id: str | bytes):
        return await self._client.xack(self.stream, self.group, msg_id)

    async def ping(self):
        return await self._client.ping()


class RedisWriter:
    def __init__(self, client: redis.Redis, config: Optional[RedisWriterConfig] = None):
        self.client = client
        if config is None:
            config = RedisWriterConfig(**json.loads(REDIS_WRITER_SETTINGS))
        self.stream = config.stream
    
    async def write_to_stream(self, data: dict):
        return await self.client.xadd(self.stream, fields=data)
