import json
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from time_series_forecasting.model import TimeSeriesForcasting


def split_df(
    df: pd.DataFrame, split: str, history_size: int = 90, horizon_size: int = 30
):
    """
    Create a training / validation samples
    Validation samples are the last horizon_size rows

    :param df:
    :param split:
    :param history_size:
    :param horizon_size:
    :return:
    """
    if split == "train":
        end_index = random.randint(horizon_size + 1, df.shape[0] - horizon_size)
    elif split in ["val", "test"]:
        end_index = df.shape[0]
    else:
        raise ValueError

    label_index = end_index - horizon_size
    start_index = max(0, label_index - history_size)

    history = df[start_index:label_index]
    targets = df[label_index:end_index]

    return history, targets


def pad_arr(arr: np.ndarray, expected_size: int = 90):
    """
    Pad top of array when there is not enough history
    :param arr:
    :param expected_size:
    :return:
    """
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode="edge")
    return arr


def df_to_np(df):
    arr = np.array(df)
    arr = pad_arr(arr)
    return arr


class Dataset(torch.utils.data.Dataset):
    def __init__(self, groups, grp_by, split, features, target):
        self.groups = groups
        self.grp_by = grp_by
        self.split = split
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        df = self.grp_by.get_group(group)

        src, trg = split_df(df, split=self.split)

        src = src[self.features + [self.target]]

        src = df_to_np(src)

        trg_in = trg[self.features + [f"{self.target}_lag_1"]]

        trg_in = np.array(trg_in)
        trg_out = np.array(trg[self.target])

        src = torch.tensor(src, dtype=torch.float)
        trg_in = torch.tensor(trg_in, dtype=torch.float)
        trg_out = torch.tensor(trg_out, dtype=torch.float)

        return src, trg_in, trg_out


def train(
    data_csv_path: str,
    feature_target_names_path: str,
    output_json_path: str,
    log_dir: str = "ts_logs",
    model_dir: str = "ts_models",
    batch_size: int = 32,
    epochs: int = 2000,
    horizon_size: int = 8,
):
    data = pd.read_csv(data_csv_path)

    with open(feature_target_names_path) as f:
        feature_target_names = json.load(f)

    data_train = data[~data[feature_target_names["target"]].isna()]

    grp_by_train = data_train.groupby(by=feature_target_names["group_by_key"])

    groups = list(grp_by_train.groups)

    full_groups = [
        grp for grp in groups if grp_by_train.get_group(grp).shape[0] > 2 * horizon_size
    ]

    train_data = Dataset(
        groups=full_groups,
        grp_by=grp_by_train,
        split="train",
        features=feature_target_names["features"],
        target=feature_target_names["target"],
    )
    val_data = Dataset(
        groups=full_groups,
        grp_by=grp_by_train,
        split="val",
        features=feature_target_names["features"],
        target=feature_target_names["target"],
    )

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=False,
    )

    model = TimeSeriesForcasting(
        n_encoder_inputs=len(feature_target_names["features"]) + 1,
        n_decoder_inputs=len(feature_target_names["features"]) + 1,
        lr=1e-4,
        dropout=0.1,
    )

    logger = TensorBoardLogger(
        save_dir=log_dir,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=model_dir,
        filename="ts",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)

    result_val = trainer.test(test_dataloaders=val_loader)

    output_json = {
        "val_loss": result_val[0]["test_loss"],
        "best_model_path": checkpoint_callback.best_model_path,
    }

    if output_json_path is not None:
        with open(output_json_path, "w") as f:
            json.dump(output_json, f, indent=4)

    return output_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path")
    parser.add_argument("--feature_target_names_path")
    parser.add_argument("--output_json_path", default=None)
    parser.add_argument("--log_dir")
    parser.add_argument("--model_dir")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    train(
        data_csv_path=args.data_csv_path,
        feature_target_names_path=args.feature_target_names_path,
        output_json_path=args.output_json_path,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
    )
