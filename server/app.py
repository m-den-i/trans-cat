import aiozmq.rpc
from server.models import CheckCategoryRequest, CheckCategoryResponse, ResponseError
from settings import SERVER_URL
from application.redis import RedisMessage
from application.app import check_labels, assign_label_and_get_response


class ServerHandler(aiozmq.rpc.AttrHandler):
    @aiozmq.rpc.method
    async def check_category(self, request: dict) -> list[dict]:
        try:
            req = CheckCategoryRequest.parse_obj(request)
            found, missing = await check_labels(req.msgs)
            if not found:
                return [None, ResponseError(error="not_found").dict()]
            return [CheckCategoryResponse(items=found).dict(), None]
        except Exception as ex:
            return [None, ResponseError(error=str(ex)).dict()]

    @aiozmq.rpc.method
    async def assign_label(self, request: dict) -> list[dict]:
        try:
            redis_msg = RedisMessage.parse_obj(request)
            msgs = await assign_label_and_get_response([redis_msg])
            resp = {ind: cat for ind, cat in zip(msgs[0].indexes, msgs[0].categories)}
            return [resp, None]
        except Exception as ex:
            return [None, ResponseError(error=str(ex)).dict()]


async def get_service():
    return await aiozmq.rpc.serve_rpc(
        ServerHandler(),
        bind=SERVER_URL,
    )


async def get_client():
    return await aiozmq.rpc.connect_rpc(
        connect=SERVER_URL,
    )
