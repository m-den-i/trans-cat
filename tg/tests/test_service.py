import pytest

from server.app import (CheckCategoryRequest, CheckCategoryResponse,
                        ResponseError)
from tg.redis import CardData, RedisMessage, TType

pytestmark = pytest.mark.asyncio


async def test_service(client, test_msgs):
    request = CheckCategoryRequest(msgs=[test_msgs[2].dict()])
    resp, err = await client.call.check_category(request.dict())
    assert err is None
    response = CheckCategoryResponse.parse_obj(resp)
    assert len(response.items) == 1

async def test_service__not_found(client, test_msgs):
    msg = RedisMessage(
        key="1002-0",
        operation_id=1002,
        data=CardData(
            amount="-10,60 PLN",
            operation_date="20-03-2023",
            type=TType.SETTLEMENT,
            card="Visa Test 1",
            place="Test Product",
        )
    )
    request = CheckCategoryRequest(msgs=[msg])
    resp, err = await client.call.check_category(request.dict())
    assert resp is None
    error = ResponseError.parse_obj(err)
    assert error.error == "not_found"
