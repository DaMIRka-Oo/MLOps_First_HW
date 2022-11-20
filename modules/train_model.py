from flask import Blueprint, Response, request

from feature_description import models_description
from feature_storage import dataset_storage, model_storage

from sklearn.model_selection import train_test_split
import pickle
import os

train_page = Blueprint('train_page', __name__)

@train_page.route("/train_model", methods = ["PUT"])
def train():
    received_data = request.json

    condition = 'dataset_nm' in received_data and 'model_type' in received_data
    assert condition, "You must add 'dataset_nm' and 'model_type'"

    dataset_nm = received_data["dataset_nm"]
    model_type = received_data["model_type"]

    data, target = dataset_storage(dataset_nm)
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

    location = './models/'
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

    data_train, _, target_train, _ = train_test_split(
                                            data, target, test_size = test_size,
                                            random_state = split_random_state
                                        )
    model.fit(data_train, target_train)

    pickle.dump(model, open(f'.\models\{model_nm}.pkl', 'wb'))

    return Response(status=201)

@train_page.route("/retraining", methods = ["POST"])
def retrain():
    received_data = request.json

    condition1 = 'model_nm' in received_data
    assert condition1, "You must add 'model_nm'"

    model_nm = received_data["model_nm"]

    location = './models/'
    models = os.listdir(location)

    condition2 = f"{model_nm}.pkl" in models
    assert condition2, "You must point off existing 'model_nm'"

    condition3 = 'dataset_nm' in received_data and 'model_type' in received_data
    assert condition3, "You must add 'dataset_nm' and 'model_type'"

    dataset_nm = received_data["dataset_nm"]
    model_type = received_data["model_type"]

    data, target = dataset_storage(dataset_nm)
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

    data_train, _, target_train, _ = train_test_split(
                                            data, target, test_size = test_size,
                                            random_state = split_random_state
                                        )
    model.fit(data_train, target_train)

    pickle.dump(model, open(f'.\models\{model_nm}.pkl', 'wb'))

    return Response(status=200)