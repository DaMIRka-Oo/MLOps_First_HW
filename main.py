import numpy as np

from flask import Flask, request, jsonify
from feature_description import datasets_description, models_description
from feature_storage import dataset_storage, model_storage

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from IPython.display import display
import pickle


# Create flask app
flask_app = Flask(__name__)

@flask_app.route("/feature_description", methods = ["POST"])
def show_descrition():
    datasets = datasets_description()
    models = models_description()

    try:
        received_data = request.json
    except:
        return jsonify(Datasets=list(datasets),
                        Models=list(models))

    if received_data["data_type"] == "Datasets":
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

    if not "test_size" in received_data:
        received_data["test_size"] = 0.33
    if not "split_random_state" in received_data:
        received_data["split_random_state"] = 666

    X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, test_size = received_data["test_size"],
                                            random_state = received_data["split_random_state"]
                                        )

    model.fit(X_train, y_train)
    pickle.dump(model, open(f'models\{received_data["model_nm"]}.pkl', 'wb'))

    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]
    roc_auc_train = roc_auc_score(y_train, y_pred_train)
    roc_auc_test = roc_auc_score(y_test, y_pred_test)

    message = (f"{model_type} was created with name '{received_data['model_nm']}'. "
               f"AUC on train is {roc_auc_train}. AUC on test is {roc_auc_test}")
    return message

if __name__ == "__main__":
    flask_app.run(debug=True)