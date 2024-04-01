from settings import DB_URL, DB_TABLE
import sqlalchemy as sa


engine = sa.create_engine(DB_URL)
_META_DATA = sa.MetaData()
_META_DATA.reflect(bind=engine)
db_table = _META_DATA.tables[DB_TABLE]
