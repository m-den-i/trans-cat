from pydantic import BaseModel
from application.models import ExistingResponse
from application.redis import RedisMessage


class CheckCategoryRequest(BaseModel):
    msgs: list[RedisMessage]


class ResponseError(BaseModel):
    error: str


class CheckCategoryResponse(BaseModel):
    items: list[ExistingResponse]
