from typing import TypedDict

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from application.response_models import MessageResponse, MessageResponseExisting


class BaseMessageCategory(TypedDict):
    text: str
    reply_markup: InlineKeyboardMarkup


def to_fix_category_response(msg: MessageResponse | MessageResponseExisting) -> BaseMessageCategory:
    if isinstance(msg, MessageResponseExisting):
        keyboard = [
            [
                InlineKeyboardButton(f"{index}", callback_data=index) for index in msg.found_indexes
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        return {"text": msg.message, "reply_markup": reply_markup}
    elif isinstance(msg, MessageResponse):
        keyboard = [
            [
                InlineKeyboardButton(f"{index}", callback_data=index) for index in msg.indexes
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        return {"text": msg.message, "reply_markup": reply_markup}
    raise NotImplemented()
