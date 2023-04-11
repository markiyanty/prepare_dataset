import datetime
import pickle
import sys

import scipy

from path import *
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_isw():
    df_isw = pd.read_csv(f"{DATA_FOLDER}{RAW_DATA_FOLDER}{ISW_FILE}")
    df_isw["date_datetime"] = pd.to_datetime(df_isw["date"])
    df_isw['date_tomorrow'] = df_isw['date_datetime'].apply(lambda x: x + datetime.timedelta(days=1))
    isw_data_exclude = ["id", "title", "url", "html", "text_v0", "text_v1", "lemm_text", "vector", "date_datetime",
                        "isWeekend"]
    df_isw = df_isw.drop(isw_data_exclude, axis=1)
    df_isw.to_csv(f"{DATA_FOLDER}{INTERIM_DATA_FOLDER}{PREPARED_ISW}", encoding="utf-8", index=False)


def get_isw_tf_idf_matrix():
    df_isw = pd.read_csv(f"{DATA_FOLDER}{RAW_DATA_FOLDER}{ISW_FILE}")
    df_isw["date_datetime"] = pd.to_datetime(df_isw["date"])
    df_isw['date_tomorrow'] = df_isw['date_datetime'].apply(lambda x: x + datetime.timedelta(days=1))
    df_isw.to_csv(f"{DATA_FOLDER}{INTERIM_DATA_FOLDER}{CLEANED_ISW}", encoding="utf-8", index=False)
    df_isw.drop(["date", "date_datetime"], axis=1)
    df_isw = pd.read_csv(f"{DATA_FOLDER}{INTERIM_DATA_FOLDER}{CLEANED_ISW}")
    dates = df_isw["date_tomorrow"].values.tolist()
    stemm_texts = df_isw["stemm_text"].values.tolist()
    vectorizer = TfidfVectorizer()
    print(vectorizer.transform(df_isw["stemm_text"].values.astype()))
    X = vectorizer.fit_transform(stemm_texts)
    tfidf_tokens = vectorizer.get_feature_names_out()
    result = pd.DataFrame(
        data=X.toarray(),
        index=dates,
        columns=tfidf_tokens
    )
    matrix = scipy.sparse.csr_matrix(result.values)
    with open("matrix.pkl", "wb") as file:
        pickle.dump(matrix, file)

    return result


if __name__ == "__main__":
    prepare_isw()
