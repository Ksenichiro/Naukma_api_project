import pickle
import pytz

import datetime as dt
import pandas as pd
import numpy as np

from dateutil import parser

from config import *
from modules.weather_module import get_weather_log_df
from modules.alarms_module import generate_features_dumb, event_holiday_is_near, load_states, calc_region_if_alarm_at
from modules.news_module import get_report_tfidf_vector, save_by_date

tz = pytz.timezone('Europe/Kyiv')

def get_yesterday_report(day_str):
    # Load yesterday ISW reports
    date = parser.parse(day_str) - dt.timedelta(days=1)
    while 'Error' in save_by_date(date):
        date -= dt.timedelta(days=1)
    tfidf_vector = get_report_tfidf_vector(f"./Reports/{date.strftime('%Y-%m-%d')}.html")
    return pd.concat([pd.DataFrame([day_str], columns=['date_tomorrow_datetime']), tfidf_vector], axis=1)

def main():
    dataset = pickle.load(open(f"{DATASET_FILE}.pkl", "rb"))

    last_hour = dt.datetime.now(tz).replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours=1)

    # Load weather last hour
    df = get_weather_log_df(last_hour)

    # Custom made dataset with most "important" russian hollidays
    holiday_df = pd.read_csv(HOLIDAY_DATASET, sep=';')
    holiday_df['date'] = holiday_df['date'].apply(pd.to_datetime)

    # Dedicated datetime column
    df['date_time'] = df.apply(lambda row: parser.parse(f"{row['day_datetime']}T{row['hour_datetime']}"), axis=1)
    # Additional region id
    df['region_id_int'] = df['region_id'].astype(int)
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

    # Generate time-based features
    states = load_states()
    df = generate_features_dumb(df, states, generate_current=False)
    df['event_indicator'] = df.apply(lambda row: calc_region_if_alarm_at(row['region'], states, row['date_time']), axis=1)
    df['event_simultaneous_alarms'] = df['event_indicator'].sum()

    # Normalize
    scaler = pickle.load(open(f'./{MODEL_FOLDER}/{scaler_model}_{scaler_version}.pkl', 'rb'))
    # Separate float values
    df_float_values = df[scaler.get_feature_names_out()]
    # Scale values
    df_float_values_scaled = pd.DataFrame(scaler.transform(df_float_values), columns=df_float_values.columns)
    df = pd.concat([df[['event_indicator', 'day_datetime', 'isw_date_tomorrow_datetime']], df_float_values_scaled], axis=1)

    # Add new rows to the dataset
    df['city_resolvedAddress'] = df.apply(lambda row: dataset[dataset['region_id'].values == row['region_id']]['city_resolvedAddress'].values[0], axis=1)
    df = df[dataset.columns]
    dataset = pd.concat([dataset, df], axis=0, ignore_index=True)
    dataset = pickle.load(open(f"{DATASET_FILE}.pkl", "rb"))

    # Save dataset
    pickle.dump(dataset, open(DATASET_FILE, 'wb'))

if __name__ == '__main__':
    main()
