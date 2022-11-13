import pandas as pd
from typing import Tuple, Union

from sklearn.datasets import load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from lightgbm import LGBMClassifier


iris, wine = load_iris(), load_wine()

datasets = {
    "load_iris": {
        "data": iris["data"],
        "target": iris["target"] >= 1
    },
    "load_wine": {
        "data": wine["data"],
        "target": wine["target"] >= 1
    },
}

models = {
        "LogisticRegression": {
            "model": LogisticRegression(),
            "random_state": 666,
            "C": 1,
            "max_iter": 100
        },
        "SVC": {
            "model": SVC(),
            "random_state": 666,
            "C": 3,
            "kernel": 'linear',
            "probability": True
        },
        "LGBMClassifier": {
            "model": LGBMClassifier(),
            "random_state": 666,
            "n_estimators": 100,
            "num_leaves": 4,
            "learning_rate": 0.01
        },
    }

def dataset_storage(dataset_nm: str) -> Tuple[pd.DataFrame, pd.Series]:
    return datasets[dataset_nm]["data"], datasets[dataset_nm]["target"]

def model_storage(model_type: str) -> Tuple[Union[LogisticRegression, SVC, LGBMClassifier], dict]:
    model_data = models[model_type].copy()
    model = model_data["model"]
    del model_data["model"]

    return model, model_data