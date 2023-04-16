from dataclasses import dataclass
import json
import os
from typing import Optional, Sequence
import redis.asyncio as redis


Redis = redis.Redis



def messages_from_read_resp(resp) -> Sequence[tuple[bytes, bytes]]:
    if len(resp):
        return resp[0][1]
    return []


async def init_redis() -> Redis:
    return await redis.from_url(os.environ["REDIS_URL"])


@dataclass
class RedisWriterConfig:
    stream: str


@dataclass
class RedisReaderConfig:
    stream: str
    group: str
    consumer: str


class RedisReader:
    def __init__(self, client: Redis, config: Optional[RedisReaderConfig] = None):
        self.client = client
        if config is None:
            config = RedisReaderConfig(**json.loads(os.environ["REDIS_READER_SETTINGS"]))
        self.stream = config.stream
        self.group = config.group
        self.consumer = config.consumer
    
    async def close(self):
        await self.client.close()

    async def read_messages(self, index: str, count: int = 1):
        resp = await self.client.xread({self.stream: index}, count=count)
        return messages_from_read_resp(resp)

    async def read_group_messages(self, index: str = ">", count: int = 1):
        resp = await self.client.xreadgroup(self.group, self.consumer, streams={self.stream: index}, count=count)
        return messages_from_read_resp(resp)

    async def read_group_pending(self, count: int = 1) -> Sequence[tuple[bytes, bytes]]:
        msgs = []
        resp = await self.client.xpending_range(self.stream, self.group, "-", "+", count, self.consumer)
        for msg in resp:
            res_ = await self.client.xrange(self.stream, msg["message_id"], "+", count=1)
            msgs.append(res_[0])
        return msgs
    
    async def ack(self, msg_id: str | bytes):
        return await self.client.xack(self.stream, self.group, msg_id)


class RedisWriter:
    def __init__(self, client: Redis, config: Optional[RedisWriterConfig] = None):
        self.client = client
        if config is None:
            config = RedisWriterConfig(**json.loads(os.environ["REDIS_WRITER_SETTINGS"]))
        self.stream = config.stream
    
    async def write_to_stream(self, data: dict):
        return await self.client.xadd(self.stream, fields=data)
