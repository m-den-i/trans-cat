import dataclasses as dtcls

import numpy as np
import pandas as pd

from lib import train
from lib.models import PredictionResult
from lib.storage.base import BaseTrainStorage, BaseUpdateStorage

def train_model_predict(storage: BaseTrainStorage, feature_vec: list[str]) -> PredictionResult:
    df = storage.load_train_model()
    df_pred = storage.load_data_for_prediction()
    return train.train_model_predict(df, df_pred, feature_vec)

def save_model_predict(storage: BaseUpdateStorage, feature_vec: list[str]):
    df = storage.load_train_model()
    df_pred = storage.load_predict_model()
    df_pred.insert(len(df_pred.columns), "Kategorie", feature_vec)
    df_all = pd.concat([df, df_pred])
    storage.save_train_model(df_all)
    storage.save_predict_model(df_pred)
