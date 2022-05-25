import pandas as pd
import os
from mmdaw import metrics
from ast import literal_eval
import numpy as np

import scipy.stats as ss

import seaborn as sns

import matplotlib.pyplot as plt

# matplotlib.rc('font', **{"family": "lmodern"})

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


def myplot(norm, ax, yy):

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
    y = yy
    algorithms = ["MMDAW", "AdwinK", "D3", "IBDD", "WATCH"]
    cs = sns.cubehelix_palette(5, start=0.5, rot=-0.75, gamma=1.2)
    ax.grid(visible=None, axis="y")
    for i, a in enumerate(algorithms):
        x_vals = group_idx + i * bar_width
        y_vals = data[data.algorithm == a].groupby("dataset")[y].mean()
        err = data[data.algorithm == a].groupby("dataset")[y].std()
        ax.bar(x_vals, y_vals, width=bar_width, edgecolor="white", label=a, color=cs[i])
        ax.errorbar(x_vals, y_vals, yerr=err, fmt="none", ecolor="black")

        ax.set_ylim((0, 1))
        ax.set_xticks(
            [r + bar_width * 2 for r in group_idx], best_configs.dataset.unique()
        )
        ax.set_axisbelow(True)  # This line added.


def plot_main_results():
    fig, ((ax11, ax12, ax13), (ax21, ax22, ax23), (ax31, ax32, ax33)) = plt.subplots(
        3, 3, figsize=(9, 4.7), sharex=True, sharey=True
    )

    ### F1
    myplot(1, ax11, "f1_detected_cps_at_y")
    myplot(2, ax12, "f1_detected_cps_at_y")
    myplot(4, ax13, "f1_detected_cps_at_y")
    ax11.set_ylabel("$F_1$")

    ### Precision
    myplot(1, ax21, "precision_y")
    myplot(2, ax22, "precision_y")
    myplot(4, ax23, "precision_y")
    ax21.set_ylabel("Precision")

    ### Recall
    myplot(1, ax31, "recall_y")
    myplot(2, ax32, "recall_y")
    myplot(4, ax33, "recall_y")
    ax31.set_ylabel("Recall")
    ax31.set_xlabel("$\\beta = 1$")
    ax32.set_xlabel("$\\beta = 1/2$")
    ax33.set_xlabel("$\\beta = 1/4$")

    Line, Label = ax11.get_legend_handles_labels()
    fig.legend(Line, Label, loc="upper center", bbox_to_anchor=(0.526, 1.04), ncol=5)
    plt.tight_layout()

    plt.savefig("figures/results.png", bbox_inches="tight")
    plt.show()


plot_main_results()
