import requests
from pathlib import Path
import json
import time
import os
from tqdm import tqdm

if __name__ == "__main__":

    url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia.org/all-access/%s/%s/all-days"
    config_path = Path(__file__).resolve().parent / "config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    base_out_path = Path(os.path.expanduser(config["top_pages_path"]))

    headers = {"accept": "application/json", "User-Agent": config["User-Agent"]}

    queries = [
        (year, str(month).zfill(2))
        for year in range(2009, 2021)
        for month in range(1, 13)
    ]

    for year, month in tqdm(queries):

        file_name = "year_%s_month_%s.json" % (year, month)
        query_url = url % (year, month)

        response = requests.get(query_url, headers=headers)

        data = response.json()

        with open(base_out_path / file_name, "w") as f:
            json.dump(data, f, indent=4)

        time.sleep(1 / 100)  # Rate limiting
