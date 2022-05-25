import pandas as pd
import os
from changeds import metrics
from ast import literal_eval
import matplotlib
import numpy as np

import scipy.stats as ss
import Orange

import seaborn as sns

import matplotlib.pyplot as plt


result_folder_name = "results"
folder = "2022-05-11"

files = os.listdir(result_folder_name + "/" + folder)

df = pd.DataFrame()

for f in files:
    df = pd.concat(
        [df, pd.read_csv(result_folder_name + "/" + folder + "/" + f, index_col=0)]
    )

df = df.reset_index(drop=True)
df = df.drop(df[df["timeout"] == True].index)
df = df.drop("timeout", axis=1)
df = df.replace({"GasSensors": "Gas", "CIFAR10": "CIFAR"})

for col in ["actual_cps", "detected_cps_at", "detected_cps"]:
    df.loc[:, col] = df.loc[:, col].apply(lambda x: literal_eval(x))

df = df.fillna(0)
df.loc[:, "percent_changes_detected"] = df.apply(
    lambda x: metrics.percent_changes_detected(x.actual_cps, x.detected_cps_at), axis=1
)


def myplot(norm, ax, y):
    Ts = {
        "MNIST": 7000 / norm,
        "CIFAR": 6000 / norm,
        "FMNIST": 7000 / norm,
        "Gas": 1159 * 2 / norm,
        "HAR": 858 * 2 / norm,
    }

    for k, v in Ts.items():
        df.loc[df["dataset"] == k, "f1_detected_cps_at"] = df.apply(
            lambda x: metrics.fb_score(x.actual_cps, x.detected_cps_at, T=v, beta=1),
            axis=1,
        )
        df.loc[df["dataset"] == k, "precision"] = df.apply(
            lambda x: metrics.prec_full(x.actual_cps, x.detected_cps_at, T=v), axis=1
        )
        df.loc[df["dataset"] == k, "recall"] = df.apply(
            lambda x: metrics.rec_full(x.actual_cps, x.detected_cps_at, T=v), axis=1
        )

    avg_results = (
        df.groupby(["dataset", "algorithm", "config"]).mean().reset_index().fillna(0)
    )
    best_configs = avg_results.loc[
        avg_results.groupby(["dataset", "algorithm"])["f1_detected_cps_at"].idxmax()
    ]
    data = pd.merge(best_configs, df, how="left", on=["dataset", "algorithm", "config"])

    data["mean_until_detection"] = data.apply(
        lambda x: metrics.mean_until_detection(x.actual_cps, x.detected_cps_at), axis=1
    )

    bar_width = 0.18

    group_idx = np.arange(5)

    algorithms = ["MMDAW", "AdwinK", "D3", "IBDD", "WATCH"]
    cs = sns.cubehelix_palette(5, start=0.5, rot=-0.75, gamma=1.2)
    ax.grid(visible=None, axis="y")
    for i, a in enumerate(algorithms):
        x_vals = group_idx + i * bar_width
        y_vals = data[data.algorithm == a].groupby("dataset")[y].mean()
        err = data[data.algorithm == a].groupby("dataset")[y].std()
        ax.bar(x_vals, y_vals, width=bar_width, edgecolor="white", label=a, color=cs[i])
        ax.errorbar(x_vals, y_vals, yerr=err, fmt="none", ecolor="black")

        # ax.set_ylim((0,1))
    ax.set_xticks([r + bar_width * 2 for r in group_idx], best_configs.dataset.unique())
    ax.set_axisbelow(True)  # This line added.


fig, (ax1, ax2) = plt.subplots(
    nrows=1, ncols=2, figsize=(9, 1.7), sharex=True, sharey=False
)

myplot(1, ax1, y="percent_changes_detected_y")
ax1.axhline(100, linestyle="--", color="black")
ax1.set_ylabel("\\% Changes Detected")
ax1.set_yscale("log")
myplot(4, ax2, y="mean_until_detection")
ax2.set_ylabel("MTD")
ax2.set_ylim((0, 2500))


Line, Label = ax1.get_legend_handles_labels()
fig.legend(Line, Label, loc="upper center", bbox_to_anchor=(0.525, 1.14), ncol=5)
plt.tight_layout()

plt.savefig("figures/percent_changes_detected.png", bbox_inches="tight")
plt.show()
