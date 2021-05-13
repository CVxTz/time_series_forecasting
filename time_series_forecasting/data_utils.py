import json
from typing import List
import pandas as pd
from pathlib import Path
import numpy as np


def add_date_cols(dataframe: pd.DataFrame, date_col: str = "timestamp"):
    """
    add time features like month, week of the year ...
    :param dataframe:
    :param date_col:
    :return:
    """

    dataframe[date_col] = pd.to_datetime(dataframe[date_col], format="%Y-%m-%d")

    dataframe["day_of_month"] = dataframe[date_col].dt.day / 31
    dataframe["day_of_year"] = dataframe[date_col].dt.dayofyear / 365
    dataframe["month"] = dataframe[date_col].dt.month / 12
    dataframe["week_of_year"] = dataframe[date_col].dt.isocalendar().week / 53
    dataframe["year"] = (dataframe[date_col].dt.year - 2015) / 5

    return dataframe, ["day_of_month", "day_of_year", "month", "week_of_year", "year"]


def add_basic_lag_features(
    dataframe: pd.DataFrame, group_by_cols: List, col_names: List, horizons: List, fill_na=True
):
    """
    Computes simple lag features
    :param dataframe:
    :param group_by_cols:
    :param col_names:
    :param horizons:
    :param fill_na:
    :return:
    """
    group_by_data = dataframe.groupby(by=group_by_cols)

    new_cols = []

    for horizon in horizons:
        dataframe[[a + "_lag_%s" % horizon for a in col_names]] = group_by_data[
            col_names
        ].shift(periods=horizon)
        new_cols += [a + "_lag_%s" % horizon for a in col_names]

    if fill_na:
        dataframe[new_cols] = dataframe[new_cols].fillna(0)

    return dataframe, new_cols


def process_df(dataframe: pd.DataFrame, target_col: str = "views"):

    """
    :param dataframe:
    :param target_col:
    :return:
    """

    dataframe, new_cols = add_date_cols(dataframe, date_col="timestamp")
    dataframe, lag_cols = add_basic_lag_features(
        dataframe, group_by_cols=["article"], col_names=[target_col], horizons=[1]
    )

    return dataframe, new_cols


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path")
    parser.add_argument("--out_path")
    parser.add_argument("--config_path")
    args = parser.parse_args()

    data = pd.read_csv(args.csv_path)

    data, cols = process_df(data)

    data.to_csv(args.out_path, index=False)

    config = {
        "features": cols,
        "target": "views",
        "group_by_key": "article",
        "lag_features": ["views_lag_1"],
    }

    with open(args.config_path, "w") as f:
        json.dump(config, f, indent=4)
