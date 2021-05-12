import matplotlib.pyplot as plt
import json
import os

import numpy as np

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        default="~/ML_DATA/wikipedia_page_views/page_views/Angelina_Jolie.json",
    )
    args = parser.parse_args()

    with open(os.path.expanduser(args.json_path)) as f:
        data = json.load(f)

    views = [np.log(x["views"] + 1) for x in data["items"]]

    print(len(views))

    views = views[-400:]

    fig = plt.figure()
    plt.plot(range(len(views)), views)
    plt.show()
