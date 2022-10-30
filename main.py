from flask import Flask, request, jsonify
from feature_description import datasets_description, models_description

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

if __name__ == "__main__":
    flask_app.run(debug=True)