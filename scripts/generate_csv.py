import pandas as pd
from pathlib import Path
import json
import os
from tqdm import tqdm


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

    df.sort_values(by=["article", "timestamp"], inplace=True)

    df.to_csv(data_path / "data.csv", index=False)
