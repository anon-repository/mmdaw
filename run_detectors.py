from time import time, strftime, gmtime
from mmdaw_adapter import MMDAWAdapter

import pathlib
from sklearn import preprocessing
import pandas as pd
from copy import deepcopy, copy
import uuid
from mmdaw.baselines import WATCH, IBDD, D3, AdwinK
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import numpy as np
from itertools import permutations
from datasets import ChangeStream
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import os

def preprocess(x):
    return preprocessing.minmax_scale(x)


class Task:
    def __init__(
        self,
        task_id,
        algorithm,
        configuration,
        dataset,
        output="results",
        timeout=1 * 60,  # maximal one minute per element
        warm_start=100,
    ):
        self.task_id = task_id
        self.algorithm = algorithm
        self.configuration = configuration
        self.dataset = copy(dataset)
        self.output = output
        self.timeout = timeout
        self.warm_start = warm_start

    def run(self):
        print(f"Run: {self.task_id}")
        result_name = self.output + "/" + str(uuid.uuid4())

        detector = self.algorithm(**self.configuration)

        # warm start

        if self.warm_start > 0:
            pre_train_dat = np.array(
                [self.dataset.next_sample()[0] for _ in range(self.warm_start)]
            ).squeeze(1)
            detector.pre_train(pre_train_dat)

            # execution
        actual_cps = []
        detected_cps = []
        detected_cps_at = (
            []
        )  # wir wollen einmal wissen, welches Element wir eingefügt haben, als ein Change erkannt wurde und zudem, wann der Change war
        i = 0
        started_at = time()
        while self.dataset.has_more_samples():
            start_time = time()
            next_sample, _, is_change = self.dataset.next_sample()
            if is_change:
                actual_cps += [i]
            detector.add_element(next_sample)
            if detector.detected_change():
                detected_cps_at += [i]
                if detector.delay:
                    detected_cps += [i - detector.delay]
            i += 1
            end_time = time()
            if end_time - start_time >= self.timeout:
                result = {
                    "algorithm": [detector.name()],
                    "config": [detector.parameter_str()],
                    "dataset": [self.dataset.id()],
                    "actual_cps": [actual_cps],
                    "detected_cps": [detected_cps],
                    "detected_cps_at": [detected_cps_at],
                    "timeout": [True],
                }
                df = pd.DataFrame.from_dict(result)
                df.to_csv(result_name)
                print(f"Aborting {self.task_id}")
                return

        result = {
            "algorithm": [detector.name()],
            "config": [detector.parameter_str()],
            "dataset": [self.dataset.id()],
            "actual_cps": [actual_cps],
            "detected_cps": [detected_cps],
            "detected_cps_at": [detected_cps_at],
            "timeout": [False],
            "runtime": [time() - started_at],
        }

        df = pd.DataFrame.from_dict(result)
        df.to_csv(result_name)


class Experiment:
    def __init__(self, configurations, datasets, reps):
        self.configurations = configurations
        self.datasets = datasets
        self.reps = reps
        foldertime = strftime("%Y-%m-%d", gmtime())
        self.output = pathlib.Path("results/" + foldertime)
        self.output.mkdir(parents=True, exist_ok=True)

    def generate_tasks(self):
        tasks = []
        task_id = 1
        for i in range(self.reps):
            for ds in self.datasets:
                for k, v in self.configurations.items():
                    for config in v:

                        t = Task(
                            task_id=task_id,
                            algorithm=k,
                            configuration=config,
                            dataset=ds,
                            output=str(self.output),
                        )
                        tasks.append(t)
                        task_id += 1
        return tasks


def get_perm_for_cd(df):
    rng = np.random.default_rng()
    classes = sorted(df["Class"].unique())
    perms = list(permutations(classes, len(classes)))
    use_perm = rng.integers(len(perms))
    mapping = dict(zip(classes, list(perms)[use_perm]))
    df = df.sort_values(
        "Class", key=lambda series: series.apply(lambda x: mapping[x])
    ).reset_index(drop=True)
    return df


class GasSensors(ChangeStream):
    def __init__(self, preprocess=None, max_len=None):
        df = pd.read_csv("./data/gas-drift_csv.csv")
        df = get_perm_for_cd(df)
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(GasSensors, self).__init__(data=data, y=np.array(y))

    def id(self) -> str:
        return "GasSensors"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class MNIST(ChangeStream):
    def __init__(self, preprocess=None, max_len=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(
            x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2])
        )
        x_test = np.reshape(
            x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2])
        )
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])
        df = pd.DataFrame(x)
        df["Class"] = y
        df = get_perm_for_cd(df)
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(MNIST, self).__init__(data=data, y=np.array(y))

    def id(self) -> str:
        return "MNIST"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class FashionMNIST(ChangeStream):
    def __init__(self, preprocess=None, max_len=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = np.reshape(
            x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2])
        )
        x_test = np.reshape(
            x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2])
        )
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])

        df = pd.DataFrame(x)
        df["Class"] = y
        df = get_perm_for_cd(df)
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(FashionMNIST, self).__init__(data=data, y=y)

    def id(self) -> str:
        return "FMNIST"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class HAR(ChangeStream):
    def __init__(self, preprocess=None, max_len=None):
        har_data_dir = "data/har"
        test = pd.read_csv(os.path.join(har_data_dir, "test.csv"))
        train = pd.read_csv(os.path.join(har_data_dir, "train.csv"))
        x = pd.concat([test, train])
        x = x.sort_values(by="Activity")
        y = LabelEncoder().fit_transform(x["Activity"])
        x = x.drop(["Activity", "subject"], axis=1)

        df = pd.DataFrame(x)
        df["Class"] = y

        df = get_perm_for_cd(df)
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)

        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(HAR, self).__init__(data=data, y=y)

    def id(self) -> str:
        return "HAR"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class CIFAR10(ChangeStream):
    def __init__(self, preprocess=None, max_len=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.dot([0.299, 0.587, 0.114])
        x_test = x_test.dot([0.299, 0.587, 0.114])
        x_train = np.reshape(
            x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2])
        )
        x_test = np.reshape(
            x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2])
        )
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train.reshape(-1), y_test.reshape(-1)])

        df = pd.DataFrame(x)
        df["Class"] = y
        df = get_perm_for_cd(df)
        data = df.drop("Class", axis=1).to_numpy()
        y = df["Class"].to_numpy()

        if max_len:
            data = data[:max_len]
            y = y[:max_len]
        if preprocess:
            data = preprocess(data)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(CIFAR10, self).__init__(data=data, y=y)

    def id(self) -> str:
        return "CIFAR10"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


if __name__ == "__main__":
    parameter_choices = {
        MMDAWAdapter: {
            "gamma": [1],
            "alpha": [
                1e-3,
                1e-2,
                1e-1,
                0.2,
            ],
        },
        AdwinK: {"k": [0.01, 0.02, 0.05, 0.1, 0.2], "delta": [0.05]},
        WATCH: {
            "kappa": [100],
            "mu": [1000, 2000],
            "epsilon": [2, 3],
            "omega": [500, 1000],
        },
        IBDD: {
            "w": [100, 200, 300],
            "m": [10, 20, 50, 100],
        },  # already tuned manually... other values work very bad.
        D3: {
            "w": [100, 200, 500],
            "roh": [0.1, 0.3, 0.5],
            "tau": [0.7, 0.8, 0.9],
            "tree_depth": [1],
        },  # tree_depths > 1 are too sensitive...
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg]))
        for alg in parameter_choices
    }

    max_len = None
    n_reps = 1

    datasets = [ # uncomment to run with different data sets
        GasSensors(preprocess=preprocess, max_len=max_len),
        #MNIST(preprocess=preprocess, max_len=max_len),
        #FashionMNIST(preprocess=preprocess, max_len=max_len),
        #HAR(preprocess=preprocess, max_len=max_len),
        #CIFAR10(preprocess=preprocess, max_len=max_len)
    ]

    ex = Experiment(algorithms, datasets, reps=n_reps)

    tasks = ex.generate_tasks()
    start_time = time()
    print(f"Total tasks: {len(tasks)}")
    Parallel(n_jobs=-2)(delayed(Task.run)(t) for t in tasks)
    end_time = time()
    print(f"Total runtime: {end_time-start_time}")
