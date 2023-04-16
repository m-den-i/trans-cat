import abc
import codecs

import chardet
import pandas as pd

from lib.storage.base import BaseTrainStorage, BaseUpdateStorage


class CSVBaseTrainStorage(BaseTrainStorage, abc.ABC):
    def __init__(self, train_path) -> None:
        self.train_path = train_path

    def load_train_model(self):
        return pd.read_csv(self.train_path, delimiter=',')


class CSVTrainStorage(CSVBaseTrainStorage):
    BLOCKSIZE = 1048576

    def __init__(self, tmp_file_data_path: str, tmp_name: str, train_path_name: str, line_start: int, line_sep: str) -> None:
        super().__init__(train_path_name)
        self.tmp_file_data_path = tmp_file_data_path
        self.tmp_name = tmp_name
        self.line_start = line_start
        self.line_sep = line_sep

    def prepare_file(self):
        with open(self.tmp_file_data_path,"rb") as f:
            tmp_file_data_codec = chardet.detect(f.read())
        with codecs.open(self.tmp_file_data_path, "r", tmp_file_data_codec['encoding']) as sourceFile:
            with codecs.open(self.tmp_name , "w", "utf-8") as targetFile:
                while True:
                    contents = sourceFile.read(self.BLOCKSIZE)
                    if not contents:
                        break
                    targetFile.write(contents)
        with open(self.tmp_name, 'r') as fin:
            data = fin.read().splitlines(True)
        with open(self.tmp_name, 'w') as fout:
            fout.writelines( [ e for e in data[self.line_start:] ])
        asdd = pd.read_csv(self.tmp_name, sep=self.line_sep)
        asdd.drop(asdd.filter(regex="Unname"),axis=1, inplace=True)
        asdd.to_csv(self.tmp_name, index=False)

    def load_data_for_prediction(self):
        self.prepare_file()
        return pd.read_csv(self.tmp_name, delimiter=',')
    
    def load_train_model(self):
        return pd.read_csv(self.train_path_name, delimiter=',')


class CSVUpdateStorage(BaseUpdateStorage):
    def __init__(self, train_path_name: str, predict_model_name: str) -> None:
        self.train_path_name = train_path_name
        self.predict_model_name = predict_model_name

    def load_train_model(self):
        return pd.read_csv(self.train_path_name, delimiter=',')

    def load_predict_model(self):
        return pd.read_csv(self.predict_model_name, delimiter=',')
    
    def save_train_model(self, df: pd.DataFrame):    
        df.to_csv(self.train_path_name, sep=",", index=False)
        
    def save_predict_model(self, df: pd.DataFrame):
        df.to_csv(self.predict_model_name, sep=",", index=False, encoding='utf-8-sig')
