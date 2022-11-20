from sklearn.datasets import load_iris, load_wine


def datasets_description():
    iris, wine = load_iris(), load_wine()

    datasets = {
        "load_iris": {
            "features_names": iris.feature_names,
            "target_names": list(iris.target_names),
            "shape": iris["data"].shape
        },
        "load_wine": {
            "features_names": wine.feature_names,
            "target_names": list(wine.target_names),
            "shape": wine["data"].shape
        }
    }

    return datasets


def models_description():
    models = {
        "LogisticRegression": {
            "random_state": 666,
            "C": 1,
            "max_iter": 100
        },
        "SVC": {
            "random_state": 666,
            "C": 3,
            "kernel": 'linear',
        },
        "LGBMClassifier": {
            "random_state": 666,
            "n_estimators": 100,
            "num_leaves": 4,
            "learning_rate": 0.01
        }
    }

    return models
