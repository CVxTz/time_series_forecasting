import pandas as pd
from typing import List, Tuple, Optional


def add_date_cols(dataframe: pd.DataFrame, date_col: str = "timestamp"):
    """
    add time features like month, week of the year ...
    :param dataframe:
    :param date_col:
    :return:
    """

    dataframe[date_col] = pd.to_datetime(dataframe[date_col], format='%Y%m%d')

    dataframe["day_of_month"] = dataframe[date_col].dt.day / 31
    dataframe["day_of_year"] = dataframe[date_col].dt.dayofyear / 365
    dataframe["month"] = dataframe[date_col].dt.month / 12
    dataframe["week_of_year"] = dataframe[date_col].dt.isocalendar().week / 53
    dataframe["year"] = (dataframe[date_col].dt.year - 2015) / 5

    return dataframe, ["day_of_month", "day_of_year", "month", "week_of_year", "year"]


def add_basic_lag_features(
    data: pd.DataFrame, group_by_cols: List, col_names: List, horizons: List
):
    """
    Computes simple lag features
    :param data:
    :param group_by_cols:
    :param col_names:
    :param horizons:
    :return:
    """
    group_by_data = data.groupby(by=group_by_cols)

    new_cols = []

    for horizon in horizons:
        data[[a + "_lag_%s" % horizon for a in col_names]] = group_by_data[
            col_names
        ].shift(periods=horizon)
        new_cols += [a + "_lag_%s" % horizon for a in col_names]

    return data, new_cols
