from pathlib import Path
import json
import os

import pandas as pd


config_path = Path(__file__).resolve().parent / "config.json"
data_path = Path(__file__).resolve().parents[1] / "data"
print(data_path)

with open(config_path, "r") as f:
    config = json.load(f)

base_in_path = Path(os.path.expanduser(config["top_pages_path"]))

all_articles = []

for json_path in base_in_path.glob("*.json"):

    with open(json_path, "r") as f:
        data = json.load(f)

        if "items" in data:
            all_articles += data["items"][0]["articles"]

df = pd.DataFrame(all_articles)

df = df.groupby("article")["views"].agg("sum").reset_index()

df.sort_values(by="views", inplace=True, ascending=False)

df.to_csv(data_path / "list_articles.csv", index=False)

with open(data_path / "list_articles.txt", 'w') as f:
    f.write("\n".join(df.article.tolist()))
