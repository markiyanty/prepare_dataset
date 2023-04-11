from path import *
import pandas as pd


def prepare_events():
    df_events = pd.read_csv(f"{DATA_FOLDER}{RAW_DATA_FOLDER}{ALARM_FILE}", sep=";")
    df_events = df_events.drop(["id", "intersection_alarm_id", "all_region", "clean_end"], axis=1)
    df_events["duration"] = pd.to_datetime(df_events["end"]) - pd.to_datetime(df_events["start"])

    df_events["start"] = pd.to_datetime(df_events["start"])
    df_events["end"] = pd.to_datetime(df_events["end"])
    df_events["duration"] = df_events["duration"].apply(lambda x: x.seconds / 60)
    df_events["start_hour"] = df_events["start"].dt.floor('H')
    df_events["end_hour"] = df_events["end"].dt.ceil('H')

    df_events = df_events.drop(["start", "end"], axis=1)
    regions_id = set(df_events["region_id"])
    df_events.to_csv(f"{DATA_FOLDER}{INTERIM_DATA_FOLDER}{PREPARED_EVENTS}", encoding="utf-8", index=False)
