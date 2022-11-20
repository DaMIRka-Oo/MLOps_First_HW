from flask import Blueprint, request, jsonify
from feature_description import datasets_description, models_description

import os

show_page = Blueprint('show_page', __name__)

@show_page.route("/feature_description", methods = ["GET", "POST"])
def show_descrition():
    datasets = datasets_description()
    models = models_description()

    if request.method == "GET":
        return jsonify(Datasets=list(datasets),
                       Models=list(models))

    if request.method == "POST":
        received_data = request.json

        condition = 'data_type' in received_data
        assert condition, "You must add 'data_type'"

        if received_data["data_type"] == "Datasets":
            return jsonify(datasets[received_data["dataset"]])
        elif received_data["data_type"] == "Models":
            return jsonify(models[received_data["model"]])


@show_page.route("/show_models", methods = ["GET"])
def show():
    location = 'models/'

    models = os.listdir(location)
    models.remove('models')

    return jsonify({"Models": models})