import pandas as pd

from lib import train
from lib.models import PredictionResult
from lib.storage.base import BaseTrainStorage, BaseUpdateStorage

def train_model_predict(storage: BaseTrainStorage, feature_vec: list[str], category_column: str = "Kategorie") -> PredictionResult:
    df = storage.load_train_model()
    df_pred = storage.load_data_for_prediction()
    return train.train_model_predict(df, df_pred, feature_vec, category_column)

def save_model_predict(storage: BaseUpdateStorage, categories: list[str], category_column: str = "Kategorie"):
    df = storage.load_train_model()
    df_pred = storage.load_predict_model()
    df_pred.insert(len(df_pred.columns), category_column, categories)
    df_all = pd.concat([df[~df.index.isin(df_pred.index)], df_pred])
    storage.save_train_model(df_all)
    storage.save_predict_model(df_pred)
