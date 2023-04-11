from path import *
import pandas as pd


def prepare_weather():
    df_weather = pd.read_csv(f"{DATA_FOLDER}{RAW_DATA_FOLDER}{WEATHER_FILE}")
    df_weather["day_datetime"] = pd.to_datetime(df_weather["day_datetime"])
    weather_exclude = [
        "day_feelslikemax",
        "day_feelslikemin",
        "day_sunriseEpoch",
        "day_sunsetEpoch",
        "day_description",
        "city_latitude",
        "city_longitude",
        "city_address",
        "city_timezone",
        "city_tzoffset",
        "day_feelslike",
        "day_precipprob",
        "day_snow",
        "day_snowdepth",
        "day_windgust",
        "day_windspeed",
        "day_winddir",
        "day_pressure",
        "day_cloudcover",
        "day_visibility",
        "day_severerisk",
        "day_conditions",
        "day_icon",
        "day_source",
        "day_preciptype",
        "day_stations",
        "hour_icon",
        "hour_source",
        "hour_stations",
        "hour_feelslike"
    ]
    df_weather_v2 = df_weather.drop(weather_exclude, axis=1)
    df_weather_v2["city"] = df_weather_v2["city_resolvedAddress"].apply(lambda x: x.split(",")[0])
    df_weather_v2["city"] = df_weather_v2["city"].replace('Хмельницька область', "Хмельницький")
    df_weather_v2.to_csv(f"{DATA_FOLDER}{INTERIM_DATA_FOLDER}{PREPARED_WEATHER}", encoding="utf-8", index=False)
