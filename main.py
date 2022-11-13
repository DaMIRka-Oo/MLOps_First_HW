import numpy as np

from flask import Flask, request, jsonify
from feature_description import datasets_description, models_description
from feature_storage import dataset_storage, model_storage

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import os
from IPython.display import display
import pickle


# Create flask app
flask_app = Flask(__name__)

@flask_app.route("/feature_description", methods = ["POST"])
def show_descrition():
    datasets = datasets_description()
    models = models_description()

    received_data = request.json

    condition = 'data_type' in received_data
    assert condition, "You must add 'data_type'"

    if received_data["data_type"] == "All":
        return jsonify(Datasets=list(datasets),
                       Models=list(models))
    elif received_data["data_type"] == "Datasets":
        return jsonify(datasets[received_data["dataset"]])
    elif received_data["data_type"] == "Models":
        return jsonify(models[received_data["model"]])

@flask_app.route("/train_model", methods = ["POST"])
def train():
    received_data = request.json

    condition = 'dataset_nm' in received_data and 'model_type' in received_data
    assert condition, "You must add 'dataset_nm' and 'model_type'"

    dataset_nm = received_data["dataset_nm"]
    model_type = received_data["model_type"]

    X, y = dataset_storage(dataset_nm)
    model, params = model_storage(model_type)

    model_descrip = models_description()[received_data["model_type"]]
    for param in model_descrip:
        if param in received_data:
            params[param] = received_data[param]
    model.set_params(**params)

    test_size, split_random_state = 0.33, 666
    if "test_size" in received_data:
        test_size = received_data["test_size"]
    if "split_random_state" in received_data:
        split_random_state = received_data["split_random_state"]

    location = 'models/'
    models = os.listdir(location)
    if not "model_nm" in received_data:
        i = 1
        while True:
            if f"model{i}.pkl" in models:
                i += 1
                continue
            break
        model_nm = f"model{i}"
    else:
        model_nm = received_data["model_nm"]
        assert f"{model_nm}.pkl" not in models, f"Model {model_nm} already exist!"

    X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, test_size = test_size,
                                            random_state = split_random_state
                                        )
    model.fit(X_train, y_train)

    pickle.dump(model, open(f'models\{model_nm}.pkl', 'wb'))

    return jsonify({"status": "ok"})

@flask_app.route("/retraining", methods = ["POST"])
def retrain():
    received_data = request.json

    condition1 = 'model_nm' in received_data
    assert condition1, "You must add 'model_nm'"

    model_nm = received_data["model_nm"]

    location = 'models/'
    models = os.listdir(location)

    condition2 = f"{model_nm}.pkl" in models
    assert condition2, "You must point off existing 'model_nm'"

    condition3 = 'dataset_nm' in received_data and 'model_type' in received_data
    assert condition3, "You must add 'dataset_nm' and 'model_type'"

    dataset_nm = received_data["dataset_nm"]
    model_type = received_data["model_type"]

    X, y = dataset_storage(dataset_nm)
    model, params = model_storage(model_type)

    model_descrip = models_description()[received_data["model_type"]]
    for param in model_descrip:
        if param in received_data:
            params[param] = received_data[param]
    model.set_params(**params)

    test_size, split_random_state = 0.33, 666
    if "test_size" in received_data:
        test_size = received_data["test_size"]
    if "split_random_state" in received_data:
        split_random_state = received_data["split_random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, test_size = test_size,
                                            random_state = split_random_state
                                        )
    model.fit(X_train, y_train)

    pickle.dump(model, open(f'models\{model_nm}.pkl', 'wb'))

    return jsonify({"status": "ok"})

@flask_app.route("/remove_models", methods = ["DELETE"])
def remove():
    received_data = request.json
    assert "remove_list" in received_data, "list of models must be by key 'remove_list'"

    location = 'models/'
    if type(received_data["remove_list"]) == str:
        if received_data["remove_list"] == 'All':
            models = os.listdir(location)
            for model in models:
                if model == 'models':
                    continue
                else:
                    file = f"{model}"
                    path = os.path.join(location, file)
                    os.remove(path)
        return jsonify({"status": "all models have droped"})

    for model in received_data["remove_list"]:
        file = f"{model}.pkl"
        path = os.path.join(location, file)
        os.remove(path)

    return jsonify({"status": f"{received_data['remove_list']} have droped"})

@flask_app.route("/show_models", methods = ["POST"])
def show():
    received_data = request.json
    assert "models_list" in received_data, "list of models must be by key 'models_list'"

    location = 'models/'
    if type(received_data["models_list"]) == str:
        if received_data["models_list"] == 'All':
            models = os.listdir(location)
            models.remove('models')

            return jsonify({"Models": models})
    return jsonify({"Models": ""})

@flask_app.route("/predict_class", methods = ["POST"])
def predict():
    received_data = request.json
    assert "model_nm" in received_data, "You must add 'model_nm'"
    assert "data" in received_data, "You must point a data"

    model_nm = received_data["model_nm"]
    data = received_data["data"]

    location = 'models/'
    models = os.listdir(location)
    assert f"{model_nm}.pkl" in models, "You must point off existing 'model_nm'"

    if "cutoff" in received_data:
        cutoff = received_data["cutoff"]
    else:
        cutoff = 0.5

    if isinstance(data, dict):
        data = list(data.values())
        data = [data]
        data = np.array(data)
    if isinstance(data, list):
        data_list = data.copy()
        data = []
        for observ in data_list:
            observ = list(observ.values())
            observ = np.array(observ)
            data.append(observ)
        data = np.array(data)

    filename = f'models/{model_nm}.pkl'
    model = pickle.load(open(filename, 'rb'))

    y_pred = model.predict_proba(data)[:, 1]
    y_pred = y_pred > cutoff
    y_pred = list(map(int, y_pred))

    return jsonify({"y_pred": y_pred})

if __name__ == "__main__":
    flask_app.run(debug=True)