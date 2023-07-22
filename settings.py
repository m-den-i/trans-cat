import os


UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER")
DOWNLOAD_FOLDER = os.environ.get("DOWNLOAD_FOLDER")
TG_BOT_KEY = os.environ["TG_BOT_KEY"]
REDIS_WRITER_SETTINGS = os.environ["REDIS_WRITER_SETTINGS"]
REDIS_READER_SETTINGS = os.environ["REDIS_READER_SETTINGS"]
REDIS_URL = os.environ["REDIS_URL"]
DB_URL = os.environ["DB_URL"]
DB_TABLE = os.environ["DB_TABLE"]
SERVER_URL = os.environ["SERVER_URL"]