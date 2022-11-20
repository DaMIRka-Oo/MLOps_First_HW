from flask import Blueprint, Response, request, jsonify
import os

remove_page = Blueprint('remove_page', __name__)

@remove_page.route("/remove_models", methods = ["DELETE"])
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

    return Response(status=200)