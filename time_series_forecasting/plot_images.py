import json

import matplotlib.pyplot as plt


if __name__ == "__main__":

    with open("data/visualization.json", "r") as f:
        data = json.load(f)

    for i, sample in enumerate(data):
        hist_size = len(sample["history"])
        gt_size = len(sample["ground_truth"])
        plt.figure()
        plt.plot(range(hist_size), sample["history"], label="input")
        plt.plot(
            range(hist_size, hist_size + gt_size), sample["ground_truth"], label="gt"
        )
        plt.plot(
            range(hist_size, hist_size + gt_size), sample["prediction"], label="pred"
        )

        plt.xlabel("Index")

        plt.ylabel("x")

        plt.legend()

        plt.savefig(f"data/images/{i}.png")
        plt.close()
