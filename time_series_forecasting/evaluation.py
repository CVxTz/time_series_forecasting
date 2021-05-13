import json
from typing import Optional
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

from time_series_forecasting.model import TimeSeriesForcasting
from time_series_forecasting.training import split_df, Dataset


def smape(true, pred):
    """
    Symmetric mean absolute percentage error
    :param true:
    :param pred:
    :return:
    """
    true = np.array(true)
    pred = np.array(pred)

    smape_val = (
        100
        / pred.size
        * np.sum(2 * (np.abs(true - pred)) / (np.abs(pred) + np.abs(true) + 1e-8))
    )

    return smape_val


def evaluate_regression(true, pred):
    """
    eval mae + smape
    :param true:
    :param pred:
    :return:
    """

    return {"smape": smape(true, pred), "mae": mean_absolute_error(true, pred)}


def evaluate(
    data_csv_path: str,
    feature_target_names_path: str,
    trained_json_path: str,
    eval_json_path: str,
    horizon_size: int = 8,
    data_for_visualization_path: Optional[str] = None,
):
    """
    Evaluates the model on the last 8 labeled weeks of the data.
    Compares the model to a simple baseline : prediction the last known value
    :param data_csv_path:
    :param feature_target_names_path:
    :param trained_json_path:
    :param eval_json_path:
    :param horizon_size:
    :param data_for_visualization_path:
    :return:
    """
    data = pd.read_csv(data_csv_path)

    with open(trained_json_path) as f:
        model_json = json.load(f)

    model_path = model_json["best_model_path"]

    with open(feature_target_names_path) as f:
        feature_target_names = json.load(f)

    target = feature_target_names["target"]

    data_train = data[~data[target].isna()]

    grp_by_train = data_train.groupby(by=feature_target_names["group_by_keys"])

    groups = list(grp_by_train.groups)

    full_groups = [
        grp for grp in groups if grp_by_train.get_group(grp).shape[0] > horizon_size
    ]

    val_data = Dataset(
        groups=full_groups,
        grp_by=grp_by_train,
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

    gt = []
    baseline_last_known_values = []
    neural_predictions = []

    data_for_visualization = []

    for i, group in enumerate(full_groups):
        time_series_data = {"history": [], "ground_truth": [], "prediction": []}

        df = grp_by_train.get_group(group)
        src, trg = split_df(df, split="val")

        time_series_data["history"] = src[target].tolist()[-60:]
        time_series_data["ground_truth"] = trg[target].tolist()

        last_known_value = src[target].values[-1]

        trg["last_known_value"] = last_known_value

        gt += trg[target].tolist()
        baseline_last_known_values += trg["last_known_value"].tolist()

        src, trg_in, _ = val_data[i]

        src, trg_in = src.unsqueeze(0), trg_in.unsqueeze(0)

        with torch.no_grad():
            prediction = model((src, trg_in[:, :1, :]))
            for j in range(1, 8):
                last_prediction = prediction[0, -1]
                trg_in[:, j, -1] = last_prediction
                prediction = model((src, trg_in[:, : (j + 1), :]))

            trg[target + "_predicted"] = (
                prediction.squeeze().numpy()
            ).tolist()

            neural_predictions += trg[target + "_predicted"].tolist()

            time_series_data["prediction"] = trg[target + "_predicted"].tolist()

        data_for_visualization.append(time_series_data)

    baseline_eval = evaluate_regression(gt, baseline_last_known_values)
    model_eval = evaluate_regression(gt, neural_predictions)

    eval_dict = {
        "Baseline_MAE": baseline_eval["mae"],
        "Baseline_SMAPE": baseline_eval["smape"],
        "Model_MAE": model_eval["mae"],
        "Model_SMAPE": model_eval["smape"],
    }

    if eval_json_path is not None:
        with open(eval_json_path, "w") as f:
            json.dump(eval_dict, f, indent=4)

    if data_for_visualization_path is not None:
        with open(data_for_visualization_path, "w") as f:
            json.dump(data_for_visualization, f, indent=4)

    for k, v in eval_dict.items():
        print(k, v)

    return eval_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path")
    parser.add_argument("--feature_target_names_path")
    parser.add_argument("--trained_json_path")
    parser.add_argument("--eval_json_path", default=None)
    parser.add_argument("--data_for_visualization_path", default=None)
    args = parser.parse_args()

    evaluate(
        data_csv_path=args.data_csv_path,
        feature_target_names_path=args.feature_target_names_path,
        trained_json_path=args.trained_json_path,
        eval_json_path=args.eval_json_path,
        data_for_visualization_path=args.data_for_visualization_path,
    )
