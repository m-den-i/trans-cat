import dataclasses as dtcls

import numpy as np

@dtcls.dataclass
class PredictionResult:
    lable_predict: list[str]
    lable_probs: np.array
    lable_predict_single: list[str]
