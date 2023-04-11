import datetime
import pickle

import scipy

from path import *
import pandas as pd
from datetime import datetime
from prepare_isw import get_isw_tf_idf_matrix


def merge_data():
    merge_weather_reg()
    merge_weather_events()
    merge_weather_event_isw()


# merge weather and regions
def merge_weather_reg():
    df_regioins = pd.read_csv(f"{DATA_FOLDER}{RAW_DATA_FOLDER}{REGIONS}")
    df_weather = pd.read_csv(f"{DATA_FOLDER}{INTERIM_DATA_FOLDER}{PREPARED_WEATHER}")
    print(df_regioins.head(5))
    df_weather_reg = pd.merge(df_weather, df_regioins, left_on="city", right_on="center_city_ua")
    df_weather_reg = df_weather_reg.drop(["city_resolvedAddress", "center_city_ua", "center_city_en", ], axis=1)
    df_weather_reg.to_csv(f"{DATA_FOLDER}{INTERIM_DATA_FOLDER}{WEATHER_REG_MERGED}", encoding="utf-8", index=False)
    print(df_weather_reg.shape)


# merge weather and events
def merge_weather_events():
    df_weather_reg = pd.read_csv(f"{DATA_FOLDER}{INTERIM_DATA_FOLDER}{WEATHER_REG_MERGED}")

    df_events = pd.read_csv(f"{DATA_FOLDER}{INTERIM_DATA_FOLDER}{PREPARED_EVENTS}")
    df_events["start_hour"] = pd.to_datetime(df_events['start_hour'])
    df_events['end_hour'] = pd.to_datetime(df_events['end_hour'])
    events_dict = df_events.to_dict('records')
    events_by_hour = []
    for event in events_dict:
        for d in pd.date_range(start=event["start_hour"], end=event["end_hour"], freq='1H'):
            et = event.copy()
            et["hour_level_event_time"] = d
            events_by_hour.append(et)

    df_events_v2 = pd.DataFrame.from_dict(events_by_hour)

    df_events_v2["hour_level_event_datetimeEpoch"] = df_events_v2["hour_level_event_time"].apply(
        lambda x: int(x.timestamp()) - 7200)

    df_weather_event = df_weather_reg.merge(df_events_v2, how="left",
                                            left_on=["region_alt", "hour_datetimeEpoch"],
                                            right_on=["region_title", "hour_level_event_datetimeEpoch"])
    df_weather_event["is_alarm"] = df_weather_event["start_hour"].apply(lambda x: True if x == x else False)

    df_weather_event["is_weekend"] = df_weather_event["day_datetime"].apply(
        lambda x: bool(datetime.strptime(x, "%Y-%m-%d").isoweekday() in [6, 7]))

    df_weather_event = df_weather_event.drop(
        ["day_datetimeEpoch", "hour_datetimeEpoch", "region_title", "region_id_y", "region_alt",
         "duration", "start_hour", "end_hour", "day_sunrise", "day_sunset", "hour_preciptype",
         "hour_solarenergy",
         "hour_level_event_datetimeEpoch", "hour_snow"], axis=1)
    df_weather_event.to_csv(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WEATHER_EVENT_MERGED}", encoding="utf-8", index=False)


def merge_weather_event_isw():
    df_weather_event = pd.read_csv(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WEATHER_EVENT_MERGED}")
    isw_data = pd.read_csv(f"{DATA_FOLDER}{INTERIM_DATA_FOLDER}{PREPARED_ISW}")
    df_weather_event_isw = df_weather_event.merge(isw_data, how="left", left_on=["day_datetime"],
                                                  right_on=["date_tomorrow"])
    df_weather_event_isw = df_weather_event_isw[df_weather_event_isw["day_datetime"] != "2022-02-24"]
    df_weather_event_isw = df_weather_event_isw.drop(["date_tomorrow", "date"], axis=1)
    df_weather_event_isw.to_csv(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WEATHER_EVENT_ISW_MERGED}", index=False,
                                encoding="utf-8")


merge_data()
