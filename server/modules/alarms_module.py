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

def __isNaN(num):
    return num != num

def load_alerts(contentType: str = "application/json"):
    # Loads list of active alerts in States

    url_base_url = "https://api.ukrainealarm.com"
    url_api = "api/v3"
    url_endpoint = "alerts"
    url_querry_params = ""

    url = f"{url_base_url}/{url_api}/{url_endpoint}?{url_querry_params}"

    payload = {}
    headers = {
        "Authorization": ALERTS_API_KEY,
        "accept": contentType
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    alarms = [alarm['activeAlerts'] for alarm in json.loads(response.text) if alarm['regionType'] == 'State']
    alarms = sum(alarms, [])
    return [alarm for alarm in alarms if alarm['type'] == 'AIR']

def load_history(regionId: int, contentType: str = "application/json"):
    # Loads last 25 alerts in given region (uses alerts-api id system)

    url_base_url = "https://api.ukrainealarm.com"
    url_api = "api/v3"
    url_endpoint = "alerts/regionHistory"
    url_querry_params = f"regionId={regionId}"

    url = f"{url_base_url}/{url_api}/{url_endpoint}?{url_querry_params}"

    payload = {}
    headers = {
        "Authorization": ALERTS_API_KEY,
        "accept": contentType
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    return [alarm for alarm in json.loads(response.text)[0]['alarms'] if alarm['alertType'] == "AIR"]

def load_states(contentType: str = "application/json"):
    # Loads info on all State regions including their names and ids

    url_base_url = "https://api.ukrainealarm.com"
    url_api = "api/v3"
    url_endpoint = "regions"
    url_querry_params = ""

    url = f"{url_base_url}/{url_api}/{url_endpoint}?{url_querry_params}"

    payload = {}
    headers = {
        "Authorization": ALERTS_API_KEY,
        "accept": contentType
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    regions_dict = json.loads(response.text)
    for state in regions_dict["states"]:
        state.pop("regionChildIds")
    return regions_dict['states']


__history_cache = {}

def __load_history(region_id: int):
    now = dt.datetime.now()
    # Load region alarms history
    if region_id in __history_cache and now - __history_cache[region_id]['timeStamp'] < dt.timedelta(minutes=30):
        # Found in cache, extract cache
        region_history = __history_cache[region_id]['history']
        return region_history
    else:
        # History for this region hasn't been downloaded yet
        region_history = load_history(region_id)
        __history_cache[region_id] = {
            'timeStamp': now,
            'history': region_history
        }
        return region_history

# Resets cache for region alarms history
def reset_cache():
    __history_cache = {}

def event_holiday_is_near(holiday_df, row):
    datetime = parser.parse(f"{row['day_datetime']} {row['hour_datetime']}")
    value = (np.abs((holiday_df['date'] - datetime) / np.timedelta64(1, 'h')) <= 3).any()
    return 1.0 if value and not __isNaN(value) else 0.0

def calc_region_alarms_history(region_name: str, states_list: list, datetime_now):
    # Get region id as in alarms api
    region_id = next((state for state in states_list if region_name in state['regionName']), None)['regionId']
    region_id = int(region_id)
    if region_id is None:
        return None
    # Get timezone and current time
    tz = pytz.timezone('Europe/Kyiv')
    now = tz.localize(datetime_now)

    region_history = __load_history(region_id)
    hours_from_last_alarm = 0.0
    alarm_count = 0.0

    # Calculate number of distinct alarms in past 24 hours
    for alarm in reversed(region_history):
        alarm_start_date = tz.localize(parser.parse(alarm['startDate']))
        if alarm_start_date > now:
                continue
        if  alarm['isContinue']:
            alarm_count += 1
        else:
            alarm_end_date = tz.localize(parser.parse(alarm['endDate']))
            if alarm_end_date > now:
                break
            if  (now - alarm_end_date) < dt.timedelta(hours=24):
                alarm_count += 1
            else:
                break
    # Calculate hours from last alarm
    last_air_alert = next((alert for alert in reversed(region_history) if tz.localize(parser.parse(alert['startDate'])) <= now), None)
    if not last_air_alert['isContinue']:
        hours_from_last_alarm = (now - tz.localize(parser.parse(last_air_alert['endDate']))).total_seconds() / 3600

    return pd.Series([alarm_count, hours_from_last_alarm])

def calc_region_if_alarm_at(region_name: str, states_list: list, date_time):
    # Calculates whether the alarm was active at the time in the region

    region_id = next((state for state in states_list if region_name in state['regionName']), None)['regionId']
    region_id = int(region_id)
    if region_id is None:
        return None
    # Get timezone and current time
    tz = pytz.timezone('Europe/Kyiv')
    date_time = tz.localize(date_time)

    region_history = __load_history(region_id)

    for alarm in reversed(region_history):
        if tz.localize(parser.parse(alarm['startDate'])) <= date_time and(alarm['isContinue'] or date_time < tz.localize(parser.parse(alarm['endDate'])) + dt.timedelta(hours=1)):
            return 1.0
        elif tz.localize(parser.parse(alarm['startDate'])) > date_time + dt.timedelta(hours=1):
            return 0.0
    return 0.0

# Returns number of alarms active now
def calc_simultaneous_alarms():
    return float(len(load_alerts()))

def generate_features_dumb(df, states = None, generate_current = True):
    # Num of separate alarms in past 24 hours

    # Load state regions metadata from alerts API
    if states is None:
        states = load_states()
    # Clear cache if any
    reset_cache()
    df[['event_alarms_past_24', 'event_hours_from_last_alarm']] = df.apply(lambda row: calc_region_alarms_history(row['region'], states, row['date_time']), axis=1)
    # Num of state regions with alarms at the moment
    if generate_current:
        df['event_simultaneous_alarms'] = calc_simultaneous_alarms()
    return df
