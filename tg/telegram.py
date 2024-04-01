#!/usr/bin/env python

# pylint: disable=unused-argument, wrong-import-position

# This program is dedicated to the public domain under the CC0 license.


"""Simple inline keyboard bot with multiple CallbackQueryHandlers.


This Bot uses the Application class to handle the bot.

First, a few callback functions are defined as callback query handler. Then, those functions are

passed to the Application and registered at their respective places.

Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:

Example of a bot that uses inline keyboard that has multiple CallbackQueryHandlers arranged in a

ConversationHandler.

Send /start to initiate the conversation.

Press Ctrl-C on the command line to stop the bot.

"""

from datetime import timedelta
import logging
from typing import Sequence

from application.app import assign_label_and_get_response
from server.app import get_service
from settings import TG_BOT_KEY

from application import app
from application.redis import RedisMessage, RedisReader, init_redis
from application.storage import SQLCategoriesUpdateStorage

from tg.models import MessageResponse, to_fix_category_response
from tg.utils import divide_chunks

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
    CallbackContext,
)


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

# Stages
START_ROUTES, FIXING, TYPING_CHOICE = range(3)

# Callback data
ONE, TWO, THREE, FOUR = range(4)


async def _load_redis(app: Application):
    logger.info("Loading Redis client")
    redis_client = await init_redis()
    redis = RedisReader(redis_client)
    app.bot_data["redis"] = redis
    assert await app.bot_data["redis"].ping()
    return redis


async def _load_redis_bg_task(app: Application):
    logger.info("Setting up Redis poll bg job")
    app.job_queue.run_repeating(poll_redis, interval=timedelta(seconds=5))


async def _load_server(app: Application):
    logger.info("Loading pyzmq server")
    server = await get_service()
    app.bot_data["server"] = server
    return server


async def load_app(app: Application):
    redis = await _load_redis(app)
    await _load_redis_bg_task(app)
    await _load_server(app)

    res = await redis.smembers("tg.financebot.members")
    for user_id in res:
        _user_id = user_id.decode()
        context = CallbackContext(app, _user_id, _user_id)
        await process_pending(context)


async def _clear_redis(app: Application):
    logger.info("Clearing Redis client.")
    redis = app.bot_data["redis"]
    await redis.close()


async def _clear_server(app: Application):
    logger.info("Clearing pyzmq server.")
    server = app.bot_data["server"]
    await server.wait_closed()


async def clear_app(app: Application):
    await _clear_redis(app)
    await _clear_server(app)


async def update_category_and_get_response(index: str, category: str, context: ContextTypes.DEFAULT_TYPE):
    await app.update_category(index, category)
    predict, result = app.get_chosen_category(index)
    return MessageResponse.from_result(predict, result)


async def process_msgs(context: ContextTypes.DEFAULT_TYPE, msgs: Sequence[RedisMessage]):
    redis: RedisReader = context.bot_data["redis"]
    response_texts = await assign_label_and_get_response(msgs)
    res = await redis.smembers("tg.financebot.members")
    for response_text in response_texts:
        for user_id in res:
            await context.bot.send_message(user_id.decode(), **to_fix_category_response(response_text))
        for index in response_text.indexes:
            await redis.ack(index)


async def process_pending(context: ContextTypes.DEFAULT_TYPE):
    redis: RedisReader = context.bot_data["redis"]
    while msgs := await redis.read_group_pending():
        await process_msgs(context, msgs)


async def poll_redis(context: ContextTypes.DEFAULT_TYPE):
    redis: RedisReader = context.bot_data["redis"]
    msgs = await redis.read_group_messages()
    if len(msgs):
        await process_msgs(context, msgs)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Send message on `/start`."""

    # Get user that sent /start and log his name
    user = update.message.from_user
    logger.info("User %s started the conversation.", user.first_name)
    res = await context.bot_data["redis"].smembers("tg.financebot.members")
    if str(user.id).encode() not in res:
        raise Exception(f"Auth error: {user.id}")
    return START_ROUTES


async def fixing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Send message on fixing."""

    user = update.effective_user
    index = update.callback_query.data
    loader = SQLCategoriesUpdateStorage(index, app.engine, app.db_table.name)
    df = loader.load_train_model()
    line = df.loc[df["id"] == index]
    description = line['description'].values[0]
    categories = loader.categories
    keyboard = [
        [
            InlineKeyboardButton(cat, callback_data=f"{cat}❆{index}") for cat in cats
        ] for cats in divide_chunks(tuple(categories), 3)
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    query = update.callback_query
    context.user_data["index"] = index
    context.user_data["message"] = query.message.id
    await query.answer()
    await query.edit_message_text(
        text=f"Choose category for [{update.callback_query.data}] {description}", reply_markup=reply_markup
    )
    return FIXING


async def to_fixing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return FIXING


async def set_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    category, index = update.callback_query.data.split("❆")
    query = update.callback_query
    response_text = await update_category_and_get_response(index, category, context)
    await query.answer()
    await query.edit_message_text(**to_fix_category_response(response_text))
    return START_ROUTES


async def regular_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Ask the user for info about the selected predefined choice."""
    text = update.message.text
    index = context.user_data["index"]
    message = context.user_data["message"]
    response_text = await update_category_and_get_response(index, text, context)
    await context.bot.edit_message_text(
        chat_id=update.message.from_user.id, 
        message_id=message,
        **to_fix_category_response(response_text),
    )
    return START_ROUTES


def main() -> None:
    """Run the bot."""

    # Create the Application and pass it your bot's token.
    application = (
        Application
        .builder()
        .token(TG_BOT_KEY)
        .post_stop(clear_app)
        .post_init(load_app)
        .build()
    )


    # Setup conversation handler with the states FIRST and SECOND
    # Use the pattern parameter to pass CallbackQueries with specific
    # data pattern to the corresponding handlers.
    # ^ means "start of line/string"
    # $ means "end of line/string"
    # So ^ABC$ will only allow 'ABC'

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            START_ROUTES: [
                CallbackQueryHandler(fixing, pattern=r"^[\-\d]+$"),
                CommandHandler("fixing", to_fixing)
            ],
            FIXING: [
                CallbackQueryHandler(set_category, pattern="^.*❆.*$"),
                MessageHandler(
                    filters.TEXT & ~(filters.COMMAND | filters.Regex("^Done$")), regular_choice
                )
            ],
        },
        fallbacks=[CommandHandler("start", start)],
    )


    # Add ConversationHandler to application that will be used for handling updates
    application.add_handler(conv_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling()
