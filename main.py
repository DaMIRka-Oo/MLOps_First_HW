from flask import Flask
from show import show_page
from remove import remove_page
from predict import predict_page
from train_model import train_page

flask_app = Flask(__name__)

flask_app.register_blueprint(show_page)
flask_app.register_blueprint(remove_page)
flask_app.register_blueprint(predict_page)
flask_app.register_blueprint(train_page)

flask_app.run(debug=True)