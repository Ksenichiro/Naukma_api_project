import datetime as dt
from dateutil import parser
import json

import requests

from flask import Flask, jsonify, request

WEATHER_API_KEY=""
MY_API_TOKEN=""

app = Flask(__name__)

class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/")
def home_page():
    return "<p><h2>KMA: Forcasting service.</h2></p>"

def verify_token(json_data):
    if json_data.get("token") is None:
        raise InvalidUsage("token is required", status_code=400)

    token = json_data.get("token")

    if token != MY_API_TOKEN:
        raise InvalidUsage("wrong API token", status_code=403)

def load_weather(location: str, aggregeteHours: int, forecastDays: int, contentType: str = "json"):    
    url_base_url = "https://weather.visualcrossing.com"
    url_api = "VisualCrossingWebServices/rest/services/weatherdata"
    url_endpoint = "forecast"
    url_querry_params = f"locations={location}&key={WEATHER_API_KEY}&contentType={contentType}"

    if aggregeteHours:
        url_querry_params += f"&aggregateHours={aggregeteHours}"
    if forecastDays:
        url_querry_params += f"&forecastDays={forecastDays}"

    url = f"{url_base_url}/{url_api}/{url_endpoint}?{url_querry_params}"
    print(url)
    
    payload = {}
    headers = {"Authorization": WEATHER_API_KEY}

    response = requests.request("GET", url, headers=headers, data=payload)
    return json.loads(response.text)     

def is_date_suitable(hour, timeSpan: int):
    date = parser.parse(hour.get("datetimeStr"))
    now = dt.datetime.now(date.tzinfo)
    return now < date < now + dt.timedelta(hours=timeSpan)

def prepare_forecast_data(weather_json, location_name: str, timeSpan: int):
    location_data = weather_json.get("locations").get(location_name)
    time_forecast = location_data.get("values")
    filtered = list(filter(lambda hour: is_date_suitable(hour, timeSpan), time_forecast))
    result_data = {
        "location": location_data.get("name"),
        "timeForecast": filtered,
        "currentConditions": location_data.get("currentConditions")
    }
    return result_data
    

@app.route(
    "/content/api/v1/integration/weather-forecast",
    methods=["POST"],
)
def forecast_endpoint():
    json_data = request.get_json()
    verify_token(json_data)

    location = ""
    if json_data.get("location"):
        location = json_data.get("location")

    weather_json = load_weather(location, 1, 1)
    prepared_data = prepare_forecast_data(weather_json, location, 12)

    return prepared_data