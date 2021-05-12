import requests
from pathlib import Path
import json
import time
import os
from tqdm import tqdm


url = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/"
    "%s/daily/%s/%s"
)

config_path = Path(__file__).resolve().parent / "config.json"

data_path = Path(__file__).resolve().parents[1] / "data"

with open(data_path / "list_articles.txt") as f:
    pages = f.read().split("\n")

with open(config_path, "r") as f:
    config = json.load(f)

base_out_path = Path(os.path.expanduser(config["page_views_path"]))

start_date = config["start_date"]
end_date = config["end_date"]

headers = {"accept": "application/json", "User-Agent": config["User-Agent"]}


for page in tqdm(pages):

    file_name = f"{page}.json"
    file_name = file_name.replace("/", "_")

    query_url = url % (page, start_date, end_date)

    response = requests.get(query_url, headers=headers)

    data = response.json()

    with open(base_out_path / file_name, "w") as f:
        json.dump(data, f, indent=4)

    time.sleep(1 / 100)  # Rate limiting
