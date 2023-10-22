import flask
import dotenv
# If you want to use environment variables:

dotenv.load_dotenv(".env")

def create_app() -> flask.Flask:
    app = flask.Flask(__name__)
    return app