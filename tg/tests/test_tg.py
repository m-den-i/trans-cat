from typing import Sequence
from unittest import mock

import pytest
from application.redis import RedisMessage
from application.storage import SQLTrainStorage
from tg.telegram import process_msgs, process_pending
from tg.tests.conftest import FakeContext

pytestmark = pytest.mark.asyncio


async def test_process_msgs(tg_context: FakeContext, test_msgs: Sequence[RedisMessage], storage: SQLTrainStorage):
    await process_msgs(tg_context, test_msgs)
    assert tg_context.bot_data["redis"].ack.await_count == 3
    tg_context.bot.send_message.assert_has_calls([
        mock.call(
            "123", 
            text='[1001-0] 71,92 PLN TR-1ABC-8KAM123@KRAJOWY INTEGRATOR PŁATNOŚCI SA->62 1090 1447 0000 0001 4793 2085 <> Suvibox (48)', 
            reply_markup=mock.ANY,
        ),
        mock.call(
            "123", 
            text='[1000-0] -10,60 PLN Zabka NANO Warszawa <> Found at [2195]: Żabka Nano\n[1002-0] -10,60 PLN Zabka NANO Warszawa <> Found at [2195]: Żabka Nano', 
            reply_markup=mock.ANY,
        ),
    ])


async def test_process_pending(tg_context_redis: FakeContext, storage: SQLTrainStorage):
    await process_pending(tg_context_redis)
    assert tg_context_redis.bot_data["redis"].ack.await_count == 1
