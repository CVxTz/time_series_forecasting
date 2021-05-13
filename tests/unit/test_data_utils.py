import pandas as pd
import pytest

from time_series_forecasting.data_utils import (
    add_date_cols,
    add_basic_lag_features,
)


def data(start_date="2018-01-01"):
    df = pd.DataFrame(
        {
            "key": ["A"] * 100 + ["B"] * 200,
            "value": list(range(100)) + list(range(1000, 1200)),
            "date": pd.date_range(start_date, periods=100, freq="D").tolist()
            + pd.date_range(start_date, periods=200, freq="D").tolist(),
        }
    )
    return df


@pytest.fixture
def tr_data():
    return data("2018-01-01")


@pytest.fixture
def te_data():
    return data("2020-01-01")


def test_add_date_cols(tr_data):
    df, new_cols = add_date_cols(tr_data, date_col="date")
    assert new_cols == ["day_of_month", "day_of_year", "month", "week_of_year", "year"]


def test_add_basic_lag_features(tr_data):
    df, new_cols = add_basic_lag_features(
        tr_data,
        group_by_cols=["key"],
        col_names=["value", "date"],
        horizons=[0, 1, 2],
        fill_na=False,
    )

    diff_date_0 = (df["date_lag_0"] - df["date"]).dropna().dt.days
    diff_date_1 = (df["date_lag_1"] - df["date"]).dropna().dt.days

    diff_value_2 = (df["value_lag_2"] - df["value"]).dropna()

    assert new_cols == [
        "value_lag_0",
        "date_lag_0",
        "value_lag_1",
        "date_lag_1",
        "value_lag_2",
        "date_lag_2",
    ]

    assert (diff_date_0 == 0).all()
    assert (diff_date_1 == -1).all()
    assert (diff_value_2 == -2).all()
