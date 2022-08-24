import pathlib
import random

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

home = pathlib.Path.cwd()


class GuidedIterator:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train_model(self, ix):


    def iterate(self, iterations, ado_stop=1.0):
        for ix in range(iterations):
            auc_val = self.train_model(ix)


def main():
    data = pd.read_parquet(home / "data" / "sample.parquet")
    y = data["targets"]
    X = data.drop(columns=["targets"])
    GI = GuidedIterator(X, y)


if __name__ == "__main__":
    main()
