from dataclasses import asdict
import asyncclick as click
from tg.app import db_table, engine
from tg.constants import CATEGORY_COLUMN
from tg.models import LabelEvent
from tg.redis import RedisWriter, RedisWriterConfig, init_redis
from tg.storage import CategoriesUpdateStorage
# You can use all of click's features as per its documentation.
# Async commands are supported seamlessly; they just work.

@click.command()
@click.option("--stream", help="Stream to push marked messages.")
@click.option("--since_id", help="The id to begin with.")
async def put_into_stream(stream, since_id):
    client = await init_redis()
    writer = RedisWriter(client, config=RedisWriterConfig(stream))
    storage = CategoriesUpdateStorage(since_id, engine, db_table)
    int_index, _ = map(int, since_id.split("-"))
    train_model = storage.load_train_model()
    try:
        index = int_index
        while values := train_model.loc[train_model["id"] == f"{index}-0"][CATEGORY_COLUMN].values:
            label = values[0]
            await writer.write_to_stream(asdict(LabelEvent(f"{index}-0", label)))
            index += 1
    except KeyError:
        print(f"Sent {index - int_index} items. Last is {index - 1}-0")
