import aiozmq.rpc
from server.models import CheckCategoryRequest, CheckCategoryResponse, ResponseError
from settings import SERVER_URL
from tg.app import check_labels




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


async def get_service():
    return await aiozmq.rpc.serve_rpc(
        ServerHandler(), 
        bind=SERVER_URL,
    )


async def get_client():
    return await aiozmq.rpc.connect_rpc(
        connect=SERVER_URL,
    )
