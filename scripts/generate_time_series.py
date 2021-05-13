import random

import pandas as pd
from tqdm import tqdm
import numpy as np
from uuid import uuid4

periods = [7, 14, 28, 30]


def get_init_df():

    date_rng = pd.date_range(start="2015-01-01", end="2020-01-01", freq="D")

    dataframe = pd.DataFrame(date_rng, columns=["timestamp"])

    dataframe["index"] = range(dataframe.shape[0])

    dataframe["article"] = uuid4().hex

    return dataframe


def set_amplitude(dataframe):

    max_step = random.randint(15, 45)
    max_amplitude = random.uniform(0.1, 1)
    offset = random.uniform(-1, 1)

    amplitude = (
        dataframe["index"]
        .apply(lambda x: max_amplitude * (x % max_step) / max_step + offset)
        .values
    )

    if random.random() < 0.5:
        amplitude = amplitude[::-1]

    dataframe["amplitude"] = amplitude

    return dataframe


def set_offset(dataframe):

    max_step = random.randint(15, 45)
    max_offset = random.uniform(-1, 1)
    base_offset = random.uniform(-1, 1)

    offset = (
        dataframe["index"]
        .apply(lambda x: max_offset * np.cos(x * 2 * np.pi / max_step) + base_offset)
        .values
    )

    if random.random() < 0.5:
        offset = offset[::-1]

    dataframe["offset"] = offset

    return dataframe


def generate_time_series(dataframe):

    period = random.choice(periods)

    dataframe["views"] = dataframe.apply(
        lambda x: np.cos(x["index"] * 2 * np.pi / period) * x["amplitude"]
        + x["offset"],
        axis=1,
    )

    return dataframe


def generate_df():
    dataframe = get_init_df()
    dataframe = set_amplitude(dataframe)
    dataframe = set_offset(dataframe)
    dataframe = generate_time_series(dataframe)
    return dataframe


if __name__ == "__main__":

    dataframes = []

    for _ in tqdm(range(5000)):
        df = generate_df()

        dataframes.append(df)

    all_data = pd.concat(dataframes, ignore_index=True)

    all_data.to_csv("data/data.csv", index=False)


