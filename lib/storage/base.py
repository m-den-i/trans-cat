import abc

import pandas as pd


class BaseTrainStorage(abc.ABC):
    @abc.abstractmethod
    def load_data_for_prediction(self) -> pd.DataFrame:
        ...
    
    @abc.abstractmethod
    def load_train_model(self) -> pd.DataFrame:
        ...


class BaseUpdateStorage(abc.ABC):
    @abc.abstractmethod
    def load_train_model(self) -> pd.DataFrame:
        ...
    
    @abc.abstractmethod
    def load_predict_model(self) -> pd.DataFrame:
        ...

    @abc.abstractmethod
    def save_train_model(self, df: pd.DataFrame):
        ...
    
    @abc.abstractmethod
    def save_predict_model(self, df: pd.DataFrame):
        ...
