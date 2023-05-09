import requests
import json
import datetime
import os
import re
import pickle
import pytz

import datetime as dt
import pandas as pd
import numpy as np

from dateutil import parser

from config import *

def load_weather(location: str, contentType: str = "json"):
    # Loads weaher forecast for next 24 hours in given location

    url_base_url = "https://weather.visualcrossing.com"
    url_api = "VisualCrossingWebServices/rest/services/timeline"
    url_endpoint = f"{location}/next24hours"
    url_querry_params = f"unitGroup=metric&include=hours%2Cdays&key={WEATHER_API_KEY}&contentType={contentType}"

    url = f"{url_base_url}/{url_api}/{url_endpoint}?{url_querry_params}"

    payload = {}
    headers = {"Authorization": WEATHER_API_KEY}

    response = requests.request("GET", url, headers=headers, data=payload)
    return json.loads(response.text)

def load_weather_at(location: str, date_time: datetime.datetime, contentType: str = "json"):
    # Loads weaher forecast for next 24 hours in given location

    url_base_url = "https://weather.visualcrossing.com"
    url_api = "VisualCrossingWebServices/rest/services/timeline"
    url_endpoint = f"{location}/{date_time.strftime('%Y-%m-%dT%H:%M:%S')}"
    url_querry_params = f"unitGroup=metric&include=current&key={WEATHER_API_KEY}&contentType={contentType}"

    url = f"{url_base_url}/{url_api}/{url_endpoint}?{url_querry_params}"

    payload = {}
    headers = {"Authorization": WEATHER_API_KEY}

    response = requests.request("GET", url, headers=headers, data=payload)
    return json.loads(response.text)

def __is_date_suitable(datetime: str, tzoffset: float, time_span: int):
    # Checks if given date is withing timespan from now using tz hour offset value
    utc_datetime = pytz.utc.localize(parser.parse(datetime))
    now = dt.datetime.now(pytz.utc) + dt.timedelta(hours=tzoffset)
    return now < utc_datetime < now + dt.timedelta(hours=time_span + 1)

def prepare_weather_data(day: dict, hour: dict):
    # Creates N hours forecast from forecast API responce for given region
    hour_data = {}

    # Write nessesary data into the responce
    hour_data["day_datetime"] = day["datetime"]
    hour_data["day_tempmin"] = day["tempmin"]
    hour_data["day_tempmax"] = day["tempmax"]
    hour_data["day_temp"] = day["temp"]
    hour_data["day_dew"] = day["dew"]
    hour_data["day_humidity"] = day["humidity"]
    hour_data["day_precip"] = day["precip"]
    hour_data["day_precipcover"] = day["precipcover"]
    hour_data["day_solarradiation"] = day["solarradiation"]
    hour_data["day_solarenergy"] = day["solarenergy"]
    hour_data["day_uvindex"] = day["uvindex"]
    hour_data["day_sunrise"] = day["sunrise"]
    hour_data["day_sunset"] = day["sunset"]
    hour_data["day_moonphase"] = day["moonphase"]
    hour_data["hour_datetime"] = hour["datetime"]
    hour_data["hour_temp"] = hour["temp"]
    hour_data["hour_humidity"] = hour["humidity"]
    hour_data["hour_dew"] = hour["dew"]
    hour_data["hour_precip"] = hour["precip"]
    hour_data["hour_precipprob"] = hour["precipprob"]
    hour_data["hour_snow"] = hour["snow"]
    hour_data["hour_snowdepth"] = hour["snowdepth"]
    hour_data["hour_preciptype"] = hour["preciptype"]
    hour_data["hour_windgust"] = hour["windgust"]
    hour_data["hour_windspeed"] = hour["windspeed"]
    hour_data["hour_winddir"] = hour["winddir"]
    hour_data["hour_pressure"] = hour["pressure"]
    hour_data["hour_visibility"] = hour["visibility"]
    hour_data["hour_cloudcover"] = hour["cloudcover"]
    hour_data["hour_solarradiation"] = hour["solarradiation"]
    hour_data["hour_solarenergy"] = hour["solarenergy"]
    hour_data["hour_uvindex"] = hour["uvindex"]
    try:
        hour_data["hour_severerisk"] = hour["severerisk"]
    except:
        hour_data["hour_severerisk"] = None
    hour_data["hour_conditions"] = hour["conditions"]

    return hour_data

def __prepare_if_sutable(weather_json, time_span: int):
    # Creates N hours forecast from forecast API responce for given region
    forecast = []
    for day in weather_json["days"]:
        for hour in day["hours"]:
            hour_data = {}

            # Check if date is within timeSpan
            datetime = f"{day['datetime']} {hour['datetime']}"
            if __is_date_suitable(datetime, weather_json['tzoffset'], time_span):
                # Parse data and append
                forecast.append(prepare_weather_data(day, hour))

    return forecast

def get_weather_forecast_df(time_span: int):
    # Creates dataframe with forecast data for given number of hours
    df = []
    # Read regions DF
    df_regions = pd.read_csv(REGIONS_DATASET, sep=',')

    for index, row in df_regions.iterrows():
        # Load forecast for each region
        try:
            weather = load_weather(row['center_city_en'])
        except:
            weather = load_weather(row['center_city_en'] + '(UA)')

        # Parse forecast for next time_span hours
        weather = __prepare_if_sutable(weather, time_span)
        df_city = pd.DataFrame(weather)

        # Add nessesary metadata
        df_city['region_id'] = float(row['region_id'])
        df_city['region'] = row['region']

        df.append(df_city)
    # Unite all loadl DFs in one DF
    return pd.concat(df, axis=0, ignore_index=True)

def get_weather_log_df(date_time: dt.datetime):
    # Creates dataframe with forecast data for given number of hours
    df = []
    # Read regions DF
    df_regions = pd.read_csv(REGIONS_DATASET, sep=',')
    for index, row in df_regions.iterrows():
        # Load forecast for each region
        try:
            weather = load_weather_at(row['center_city_en'], date_time)
        except:
            weather = load_weather_at(row['center_city_en'] + '(UA)', date_time)

        # Parse forecast for next time_span hours
        weather = prepare_weather_data(weather['days'][0], weather['currentConditions'])
        df_city = pd.DataFrame([weather])

        # Add nessesary metadata
        df_city['region_id'] = float(row['region_id'])
        df_city['region'] = row['region']

        df.append(df_city)
    # Unite all loadl DFs in one DF
    return pd.concat(df, axis=0, ignore_index=True)