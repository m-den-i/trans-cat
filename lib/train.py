import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from lib.models import PredictionResult

def train_model_predict(df_model: pd.DataFrame, df_pred: pd.DataFrame, feature_vec: list[str], category_column: str = "Kategorie") -> PredictionResult:
    df_x = pd.DataFrame(df_model, columns=feature_vec).values.tolist()
    df_x = [ ' '.join(str(e) for e in st) for st in df_x ]

    df_y = pd.DataFrame(df_model, columns=[category_column]).values.tolist()
    df_y_flat = list(dict.fromkeys( [subitem for item in df_y for subitem in item] ))

    df_x_predict = pd.DataFrame(df_pred, columns=feature_vec).values.tolist()
    df_x_predict = [ ' '.join(str(e) for e in st) for st in df_x_predict ]

    vec = CountVectorizer(analyzer="word",token_pattern="[A-Z]{3,}[a-z]{3,}|[a-zA-Z]{3,}",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)

    df_x_trans = vec.fit_transform(df_x + df_x_predict)
    output_transformer = OrdinalEncoder()

    X_all = df_x_trans.toarray()

    X_train = X_all[0:len(df_x)]
    X_predict = X_all[len(df_x):]
    Y = output_transformer.fit_transform(df_y)
    clf = LogisticRegression(random_state=0, max_iter=5000)
    #clf = DecisionTreeClassifier(random_state=0)
    #clf = RandomForestClassifier(max_depth=2, random_state=0)

    clf.fit(X_train,Y)

    y_pred = clf.predict(X_predict)
    y_pred_prob = clf.predict_proba(X_predict)
    y_pred_str = output_transformer.inverse_transform(y_pred.reshape(-1, 1))
    y_pred_str = [e for ee in y_pred_str for e in ee]
    # print(y_pred_prob)
    # print( type(y_pred_prob) )

    y_pred_prob_single = 120.0 * np.amax(y_pred_prob, axis=1)
    res = PredictionResult(**{
        "lable_predict": y_pred_str,
        "lable_predict_single": [ e for e in df_y_flat],
        "lable_probs": y_pred_prob_single,
    })
    return res
