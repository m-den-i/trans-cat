from datetime import datetime
import os
import random as rand

import pandas as pd
import sqlalchemy as sa
from lib.storage.base import BaseUpdateStorage, BaseTrainStorage
from tg.constants import CATEGORY_COLUMN
from tg.models import get_df_from_response


class _BaseLoader:
    def __init__(self, engine: sa.Engine, table: str) -> None:
        self.engine = engine
        self.table = table

    def _load_train_model(self):
        with self.engine.connect() as connection:
            df = pd.read_sql_table(self.table, connection)
        df["operation_date"] = df["operation_date"].map(lambda x: datetime.strftime(x, "%d-%m-%Y"))
        df["id"] = df["id"].map(lambda x: f"{x}-0")
        return df


class CustomTrainStorage(_BaseLoader, BaseTrainStorage):
    def __init__(self, msg, engine: sa.Engine, table: str) -> None:
        assert len(msg)
        super().__init__(engine, table)
        self.df = get_df_from_response(msg)
        self.train_model = self._load_train_model()

    def load_data_for_prediction(self) -> pd.DataFrame:
        return self.df

    def load_train_model(self):
        return self.train_model


class CustomUpdateStorage(BaseUpdateStorage):
    def __init__(self, train_model: pd.DataFrame, predict_model: pd.DataFrame, engine: sa.Engine, table: str):
        self.predict_model = predict_model
        self.train_model = train_model
        self.engine = engine
        self.table = table

    def load_predict_model(self):
        return self.predict_model
    
    def load_train_model(self) -> pd.DataFrame:
        return self.train_model
    
    def save_predict_model(self, df: pd.DataFrame):
        ...

    def save_train_model(self, df: pd.DataFrame):
        with self.engine.connect() as connection:
            resp = tuple(connection.execute(sa.text(f"select id from {self.table} order by id desc limit 1;")))
            last_index = resp[0][0] if len(resp) else 0
            df["operation_date"] = df["operation_date"].map(lambda x: datetime.strptime(x, "%d-%m-%Y"))
            df["id"] = df["id"].map(lambda x: int(x.split("-")[0]))
            df = df[df["id"] > last_index]
            df.to_sql(self.table, connection, if_exists="append", index=False)
            connection.commit()
        return self.train_model


class CategoriesUpdateStorage(_BaseLoader, BaseUpdateStorage):
    def __init__(self, index: str, engine: sa.Engine, table: str):
        self.index = index
        super().__init__(engine, table)
        self.train_model = self._load_train_model()

    def save_predict_model(self, df: pd.DataFrame):
        update_rows = []
        for ind in df['id']:
            update_rows.append(
                {"id": ind.split("-")[0], "cat": df.loc[df['id'] == ind].iloc[0][CATEGORY_COLUMN]}
            )
        with self.engine.connect() as connection:
            for upd in update_rows:
                connection.execute(sa.text(f"update {self.table} set {CATEGORY_COLUMN} = :cat where id = :id"), upd)
            connection.commit()
                
    def load_train_model(self) -> pd.DataFrame:
        return self.train_model

    def load_predict_model(self) -> pd.DataFrame:
        return self.train_model.loc[self.train_model["id"] == self.index, self.train_model.columns != CATEGORY_COLUMN]

    def save_train_model(self, df: pd.DataFrame):
        ...
    
    def load_data_for_prediction(self) -> pd.DataFrame:
        return self.load_predict_model()
    
    @property
    def categories(self):
        return set(self.train_model[CATEGORY_COLUMN])
