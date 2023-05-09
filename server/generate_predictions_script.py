import pickle
import pytz
import json

import datetime as dt
import pandas as pd
import numpy as np

from dateutil import parser

from config import *
from modules.weather_module import get_weather_forecast_df
from modules.alarms_module import generate_features_dumb, event_holiday_is_near, load_states
from modules.news_module import get_report_tfidf_vector, save_by_date

def isNaN(num):
    return num != num

tz = pytz.timezone('Europe/Kyiv')

# Generate predictions file for 12 hours in all regions

def get_yesterday_report(day_str):
    # Load yesterday ISW reports
    date = parser.parse(day_str) - dt.timedelta(days=1)
    while 'Error' in save_by_date(date):
        date -= dt.timedelta(days=1)
    tfidf_vector = get_report_tfidf_vector(f"./Reports/{date.strftime('%Y-%m-%d')}.html")
    return pd.concat([pd.DataFrame([day_str], columns=['date_tomorrow_datetime']), tfidf_vector], axis=1)

def get_prediction(df, model, scaler):
    # Generate predictions

    # Normalize
    # Separate float values
    df_float_values = df[scaler.get_feature_names_out()]
    if df_float_values.to_numpy().ndim < 2:
        df_float_values = pd.DataFrame({1: df_float_values.sparse.to_dense()}).T
    # Scale values
    df_float_values_scaled = pd.DataFrame(scaler.transform(df_float_values), columns=df_float_values.columns)
    df_float_values_scaled['day_of_week'] = df['day_of_week']

    # Generate prediction
    prediction = model.predict(df_float_values_scaled)
    return prediction

def generate_features_smart(df, model, scaler):
    # Generates features based on model predictions for prevous hours

    # Generate dumb features and prediction
    df = generate_features_dumb(df)
    df['alarm_marker'] = pd.Series(get_prediction(df.copy(), model, scaler))

    # Create new column to account for predicted alarms
    df['event_alarms_past_24_modifier'] = 0
    df['was_alarm'] = df['event_hours_from_last_alarm'].apply(lambda hours: hours < 1.0)

    # Recalculate features
    hours = forecast_hours
    regions = len(df['region_id'].unique())

    # Go through each hour and update values
    for h in range(1, hours):
        marker_sum = 0.0

        # Go through each region
        for r in range(0, regions):
            df.loc[r * hours + h, 'event_alarms_past_24_modifier'] = df.loc[r * hours + h - 1, 'event_alarms_past_24_modifier'] + df.loc[r * hours + h - 1, 'alarm_marker']
            # Reset alarm cooldown timer if new alarm predicted
            df.loc[r * hours + h, 'event_hours_from_last_alarm'] = 1.0 if df.loc[r * hours + h - 1, 'alarm_marker'] == 0 and df.loc[r * hours + h, 'was_alarm'] else df.loc[r * hours + h - 1, 'event_hours_from_last_alarm'] + 1
            # Calculate num of predicted simultanious alarms
            marker_sum += df.loc[r * hours + h - 1, 'alarm_marker']
            # Save prev continuous alarm info
            df.loc[r * hours + h, 'was_alarm'] = df.loc[r * hours + h - 1, 'alarm_marker']

        # Iterate once more
        for r in range(0, regions):
            # Write calculated values
            df.loc[r * hours + h, 'event_simultaneous_alarms'] = marker_sum
            df.loc[r * hours + h, 'event_alarms_past_24'] += df.loc[r * hours + h, 'event_alarms_past_24_modifier']
            # Update prediction with updated features
            df.loc[r * hours + h, 'alarm_marker'] = get_prediction(df.iloc[r * hours + h].copy(), model, scaler)[0]

    df = df.drop(['event_alarms_past_24_modifier', 'was_alarm'], axis=1)

    return df

def main():
    # Load weather forecast for 12 hours
    df = get_weather_forecast_df(forecast_hours)

    # Custom made dataset with most "important" russian hollidays
    holiday_df = pd.read_csv(HOLIDAY_DATASET, sep=';')
    holiday_df['date'] = holiday_df['date'].apply(pd.to_datetime)

    # Dedicated datetime column
    df['date_time'] = df.apply(lambda row: parser.parse(f"{row['day_datetime']}T{row['hour_datetime']}"), axis=1)
    # Additional region id
    df['region_id_int'] = df['region_id'].astype(int)
    # Metadata as features
    df['day_of_week'] = df['date_time'].dt.dayofweek
    # If within 3 days from holiday
    df['event_holiday_is_near'] = df.apply(lambda row: event_holiday_is_near(holiday_df, row), axis=1)

    # Convet time to float
    df['day_sunset'] = df['day_sunset'].apply(lambda x:
        (parser.parse(x) - dt.datetime.strptime("00:00:00", "%H:%M:%S")).total_seconds()
    )
    df['day_sunrise'] = df['day_sunrise'].apply(lambda x:
        (parser.parse(x) - dt.datetime.strptime("00:00:00", "%H:%M:%S")).total_seconds()
    )
    df['hour_datetime'] = df['hour_datetime'].apply(lambda x:
        (parser.parse(x) - dt.datetime.strptime("00:00:00", "%H:%M:%S")).total_seconds()//3600
    )

    # Encode categorical values
    df['hour_preciptype'] = df['hour_preciptype'].apply(lambda a: str(a) if a else np.nan)
    le = pickle.load(open(f'./{MODEL_FOLDER}/{preciptype_encoder_model}_{preciptype_encoder_verssion}.pkl', 'rb'))
    df['hour_preciptype'] = le.transform(df['hour_preciptype']).astype(float)

    le = pickle.load(open(f'./{MODEL_FOLDER}/{conditions_encoder_model}_{conditions_encoder_version}.pkl', 'rb'))
    df['hour_conditions'] = le.transform(df['hour_conditions']).astype(float)

    # Fillna
    df.fillna(0.0, inplace=True)

    # Parse TF-IDF
    df_tfidf = []
    for day in df['day_datetime'].unique():
        df_tfidf.append(get_yesterday_report(day))
    df_tfidf = pd.concat(df_tfidf, axis=0, ignore_index=True)

    # Merge weather events dataset with yesterday report tfidf matrix (takes 2m to execute)
    df = df.merge(df_tfidf.add_prefix("isw_"),
                                    how="left",
                                    left_on="day_datetime",
                                    right_on="isw_date_tomorrow_datetime")

    # Load model
    model = pickle.load(open(f'./{MODEL_FOLDER}/{model_file_name}.pkl', 'rb'))
    scaler = pickle.load(open(f'./{MODEL_FOLDER}/{scaler_model}_{scaler_version}.pkl', 'rb'))

    # Generate features
    df = generate_features_smart(df, model, scaler)

    # Save prediction to .csv
    res = pd.DataFrame()
    res[['date_time', 'region_id', 'alarm_marker']] = df[['date_time', 'region_id_int', 'alarm_marker']].copy()
    res.to_csv(PREDICTIONS_FILE, sep=';')

    # Write metadata

    status = {}
    try:
        with open(STATE_FILE, 'r') as handle:
            status = json.load(handle)
    except:
        status = {}
    with open(STATE_FILE, 'w') as handle:
        status["last_prediciotn_time"] = str(dt.datetime.now(tz))
        json.dump(status, handle)


if __name__ == '__main__':
    main()
