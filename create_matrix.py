import pickle

import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit

from path import *
import pandas as pd


def separate_data():
    df_weather_event_isw = pd.read_csv(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WEATHER_EVENT_ISW_MERGED}")
    df_weather_event_isw["day_datetime"] = pd.to_datetime(df_weather_event_isw["day_datetime"])

    df_weather_event_isw.set_index('day_datetime', inplace=True)
    df_weather_event_isw.sort_index(inplace=True)
    df_weather_event_isw = df_weather_event_isw.drop(
        ["city", "region", "region_city", "hour_level_event_time",
         "hour_datetime"], axis=1)

    df_weather_event_isw["is_weekend"] = df_weather_event_isw["is_weekend"].astype(int)
    df_weather_event_isw["hour_conditions"] = df_weather_event_isw["hour_conditions"].astype('category').cat.codes
    y = df_weather_event_isw["is_alarm"].astype(int)
    X = df_weather_event_isw.drop(["is_alarm"], axis=1)
    corpus = X["stemm_text"].values.astype("U")
    X = X.drop(["stemm_text"], axis=1).fillna(method="ffill")

    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    tf_idf = vectorizer.transform(corpus)
    print("Transformed", tf_idf.shape)

    tss = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tss.split(X):
        X_train, X_test, X_train_tf_idf, X_test_tf_idf = X.iloc[train_index, :], X.iloc[test_index, :], tf_idf[
                                                                                                        train_index,
                                                                                                        :], tf_idf[
                                                                                                            test_index,
                                                                                                            :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X_train.to_csv(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{X_TRAIN}", index=False, encoding="utf-8")
    X_test.to_csv(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{X_TEST}", index=False, encoding="utf-8")
    y_test.to_csv(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{Y_TEST}", index=False, encoding="utf-8")
    y_train.to_csv(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{Y_TRAIN}", index=False, encoding="utf-8")
    with open(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{X_TEST_TFIDF_MATRIX}", "wb") as file:
        pickle.dump(X_test_tf_idf, file)
    with open(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{X_TRAIN_TFIDF_MATRIX}", "wb") as file:
        pickle.dump(X_train_tf_idf, file)


def create_matrix():
    separate_data()
    merge_matrix(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{X_TEST}", f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{X_TEST_TFIDF_MATRIX}",
                 f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WORK_TEST_MATRIX}")
    merge_matrix(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{X_TRAIN}",
                 f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{X_TRAIN_TFIDF_MATRIX}",
                 f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WORK_TRAIN_MATRIX}")
    load_y()


def load_y():
    y_test = pd.read_csv(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{Y_TEST}")
    y_test = scipy.sparse.csr_matrix(y_test.values)
    print(y_test.shape)
    with open(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WORK_Y_TEST}", "wb") as file:
        pickle.dump(y_test, file)

    y_train = pd.read_csv(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{Y_TRAIN}")
    y_train = scipy.sparse.csr_matrix(y_train.values)
    print(y_test.shape)
    with open(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WORK_Y_TRAIN}", "wb") as file:
        pickle.dump(y_train, file)
    print(y_train.shape)


def merge_matrix(frame_file_path, tf_idf_matrix_file_path, work_matrix_file_path):
    with open(tf_idf_matrix_file_path, "rb") as file:
        tf_idf = pickle.load(file)

    df_weather_event = pd.read_csv(frame_file_path)

    df_weather_event = scipy.sparse.csr_matrix(df_weather_event.values)
    print(df_weather_event)
    print(tf_idf.shape)
    print(df_weather_event.shape)
    df_features = scipy.sparse.hstack((tf_idf, df_weather_event), format="csr")
    print(df_features.shape)

    with open(work_matrix_file_path, "wb") as file:
        pickle.dump(df_features, file)