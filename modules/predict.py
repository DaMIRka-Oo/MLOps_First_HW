from flask import Blueprint, request, jsonify

import numpy as np
import pickle
import os

predict_page = Blueprint('predict_page', __name__)

@predict_page.route("/predict_class", methods = ["POST"])
def predict():
    received_data = request.json
    assert "model_nm" in received_data, "You must add 'model_nm'"
    assert "data" in received_data, "You must point a data"

    model_nm = received_data["model_nm"]
    data = received_data["data"]

    location = './models/'
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

    filename = f'./models/{model_nm}.pkl'
    model = pickle.load(open(filename, 'rb'))

    y_pred = model.predict_proba(data)[:, 1]
    y_pred = y_pred > cutoff
    y_pred = list(map(int, y_pred))

    return jsonify({"y_pred": y_pred})