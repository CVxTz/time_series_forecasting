import json

import pandas as pd
import torch

from time_series_forecasting.model import TimeSeriesForcasting
from time_series_forecasting.training import split_df, Dataset


def predict(
    data_csv_path: str,
    feature_target_names_path: str,
    trained_json_path: str,
    predict_csv_path: str,
    target_norm: float = 100.0,
):
    """
    predict the model on the last 8 weeks of the data.
    Compares the model to a simple baseline : prediction the last known value
    :param data_csv_path:
    :param feature_target_names_path:
    :param trained_json_path:
    :param predict_csv_path:
    :param target_norm:
    :return:
    """
    data = pd.read_csv(data_csv_path)

    with open(trained_json_path) as f:
        model_json = json.load(f)

    model_path = model_json["best_model_path"]

    with open(feature_target_names_path) as f:
        feature_target_names = json.load(f)

    target = feature_target_names["target"]

    grp_by = data.groupby(by=feature_target_names["group_by_keys"])

    groups = list(grp_by.groups)

    test_data = Dataset(
        groups=groups,
        grp_by=grp_by,
        split="val",
        features=feature_target_names["features"],
        target=feature_target_names["target"],
    )

    model = TimeSeriesForcasting(
        n_encoder_inputs=len(feature_target_names["features"]) + 1,
        n_decoder_inputs=len(feature_target_names["features"]) + 1,
        lr=1e-4,
        dropout=0.5,
    )
    model.load_state_dict(torch.load(model_path)["state_dict"])

    model.eval()

    neural_predictions = []

    for i, group in enumerate(groups):
        df = grp_by.get_group(group)
        src, trg = split_df(df, split="test")

        src, trg_in, _ = test_data[i]

        src, trg_in = src.unsqueeze(0), trg_in.unsqueeze(0)

        with torch.no_grad():
            prediction = model((src, trg_in[:, :1, :]))
            for j in range(1, 8):
                last_prediction = prediction[0, -1]
                trg_in[:, j, -1] = last_prediction
                prediction = model((src, trg_in[:, : (j + 1), :]))

            trg[target] = (prediction.squeeze().numpy() * target_norm).tolist()

            neural_predictions.append(trg)

    predicted_df = pd.concat(neural_predictions)

    predicted_df = predicted_df[predicted_df.split == "test"]

    predicted_df = predicted_df[
        ["day_id", "but_num_business_unit", "dpt_num_department", "turnover"]
    ]

    if predict_csv_path is not None:
        predicted_df.to_csv(predict_csv_path, index=False)

    return predicted_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path")
    parser.add_argument("--feature_target_names_path")
    parser.add_argument("--trained_json_path")
    parser.add_argument("--predict_csv_path", default=None)
    args = parser.parse_args()

    predict(
        data_csv_path=args.data_csv_path,
        feature_target_names_path=args.feature_target_names_path,
        trained_json_path=args.trained_json_path,
        predict_csv_path=args.predict_csv_path,
    )
