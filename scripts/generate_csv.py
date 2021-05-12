import pandas as pd
from pathlib import Path
import json
import os
from tqdm import tqdm


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


if __name__ == "__main__":

    config_path = Path(__file__).resolve().parent / "config.json"

    data_path = Path(__file__).resolve().parents[1] / "data"

    with open(config_path, "r") as f:
        config = json.load(f)

    base_in_path = Path(os.path.expanduser(config["page_views_path"]))

    samples = []

    for path in tqdm(base_in_path.glob("*.json")):

        with open(path, "r") as f:
            data = json.load(f)

            if "items" in data and len(data["items"]) > 500:
                samples += [
                    {
                        "article": x["article"],
                        "timestamp": x["timestamp"][:-2],
                        "views": x["views"],
                    }
                    for x in data["items"]
                ]

    df = pd.DataFrame(samples)

    df, cols = add_date_cols(df)

    df.to_csv(data_path / "data.csv", index=False)
