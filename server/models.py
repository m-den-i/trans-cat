from pydantic import BaseModel
from tg.models import ExistingResponse
from tg.redis import RedisMessage


class CheckCategoryRequest(BaseModel):
    msgs: list[RedisMessage]


class ResponseError(BaseModel):
    error: str


class CheckCategoryResponse(BaseModel):
    items: list[ExistingResponse]
