import json
import os

import matplotlib.pyplot as plt


if __name__ == "__main__":

    with open("data/visualization.json", "r") as f:
        data = json.load(f)

    os.makedirs("data/images", exist_ok=True)

    for i, sample in enumerate(data):
        hist_size = len(sample["history"])
        gt_size = len(sample["ground_truth"])
        plt.figure()
        plt.plot(range(hist_size), sample["history"], label="History")
        plt.plot(
            range(hist_size, hist_size + gt_size), sample["ground_truth"], label="Ground Truth"
        )
        plt.plot(
            range(hist_size, hist_size + gt_size), sample["prediction"], label="Prediction"
        )

        plt.xlabel("Time")

        plt.ylabel("Time Series")

        plt.legend()

        plt.savefig(f"data/images/{i}.png")
        plt.close()
