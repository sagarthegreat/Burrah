#for using the terra api and authentication of user from front to back end:
import datetime
import requests
import flask
import json
import http
import os
import logging
from flask import request
from terra import Terra


logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("app")


terra = Terra(api_key=os.environ["TERRA_API_KEY"], dev_id=os.environ["TERRA_DEV_ID"], secret=os.environ["TERRA_WEBHOOK_SECRET"])

def hello_terra():
    # Print it here
    print("hello")
    # print(json.dumps(request.get_json(), indent = 4))

    # You can directly use this to handle a terra webhook:
    # response = terra.handle_flask_webhook(request)

    return flask.Response("Yay Terra is awesome",  http.HTTPStatus.OK)


def authenticate():
    # Assuming Terra API uses an API key for authentication
    widget_response = terra.generate_widget_session(
        reference_id="USER ID IN YOUR APP",
        #providers=["GARMIN", "APPLE", "DECATHLON", "POLAR", "GOOGLE"],
        providers=['BIOSTRAP', 'CARDIOMOOD', 'CONCEPT2', 'COROS', 'CRONOMETER', 'DEXCOM', 'EATTHISMUCH', 'EIGHT',
                   'FATSECRET', 'FINALSURGE', 'FITBIT', 'FREESTYLELIBRE', 'GARMIN', 'GOOGLE', 'HAMMERHEAD', 'HUAWEI',
                   'IFIT', 'INBODY', 'KOMOOT', 'LIVEROWING', 'LEZYNE', 'MOXY', 'NUTRACHECK', 'OMRON', 'OMRONUS', 'OURA',
                   'PELOTON', 'POLAR', 'PUL', 'RENPHO', 'RIDEWITHGPS', 'ROUVY', 'SUUNTO', 'TECHNOGYM', 'TEMPO',
                   'TRIDOT', 'TREDICT', 'TRAININGPEAKS', 'TRAINASONE', 'TRAINERROAD', 'UNDERARMOUR', 'VIRTUAGYM',
                   'WAHOO', 'WEAROS', 'WHOOP', 'WITHINGS', 'XOSS', 'ZWIFT', 'XERT', 'BRYTONSPORT', 'TODAYSPLAN', 'WGER',
                   'VELOHERO', 'CYCLINGANALYTICS', 'NOLIO', 'TRAINXHALE', 'KETOMOJOUS', 'KETOMOJOEU', 'STRAVA', 'CLUE',
                   'HEALTHGAUGE', 'MACROSFIRST'],
        auth_success_redirect_url="http://localhost:8080/success",
        auth_failure_redirect_url="https://failure.url",
        language="en"
    ).get_parsed_response()

    print(widget_response)
    # Use the API key to authenticate with the Terra API
    # Depending on the API's authentication method, you may need to set headers or parameters

    # If successful, redirect to a specific endpoint or URL
    return flask.redirect(widget_response.url)

def success():
    if user_id := flask.request.args.get("user_id"):
        daily_resp = terra.get_daily_for_user(terra.from_user_id(user_id), start_date=datetime.datetime.now() - datetime.timedelta(days=2), to_webhook=False).get_parsed_response()
        daily_obj = daily_resp.data
        print(daily_obj)

    return flask.make_response("success!", 200)


def setup(app: flask.Flask):
    bp = flask.Blueprint("sample", __name__, "")
    bp.add_url_rule("/hello", view_func = hello_terra, methods = ["POST", "GET"])
    bp.add_url_rule("/auth", view_func=authenticate, methods=["POST", "GET"])
    bp.add_url_rule("/success", view_func=success, methods=["POST", "GET"])
    bp.add_url_rule("/webhook", view_func =consume_terra_webhook, methods = ["POST", "GET"])
    app.register_blueprint(bp)


def consume_terra_webhook() -> flask.Response:
    # body_str = str(request.get_data(), 'utf-8')
    body = request.get_json()
    _LOGGER.info(
        "Received webhook for user %s of type %s",
        body.get("user", {}).get("user_id"),
        body["type"])
    verified = terra.check_terra_signature(request.get_data().decode("utf-8"), request.headers['terra-signature'])
    if verified:
        return flask.Response(status=200)
    else:
        return flask.Response(status=403)


