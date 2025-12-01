import glob
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_spam_dataset(load_train_labels: bool = False, split_dev_valid: bool = False):
    """
    Loads YouTube spam dataset from /data and returns train/test
    (or train/dev/valid/test if split_dev_valid=True).
    """

    filenames = sorted(glob.glob("data/Youtube*.csv"))

    dfs = []
    for i, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename)
        df.columns = map(str.lower, df.columns)
        df = df.drop("comment_id", axis=1)
        df["video"] = [i] * len(df)
        df = df.rename(columns={"class": "label", "content": "text"})
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        dfs.append(df)

    # Collect first 4 for training pool
    df_train = pd.concat(dfs[:4], ignore_index=True)

    # Small dev sample
    df_dev = df_train.sample(100, random_state=123)

    # Remove labels for weak supervision unless asked
    if not load_train_labels:
        df_train = df_train.copy()
        df_train["label"] = -1

    # 5th file → validation + test
    df_valid_test = dfs[4]

    df_valid, df_test = train_test_split(
        df_valid_test,
        test_size=250,
        random_state=123,
        stratify=df_valid_test["label"],
    )

    df_train = df_train.reset_index(drop=True)
    df_dev = df_dev.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    if split_dev_valid:
        return df_train, df_dev, df_valid, df_test
    else:
        return df_train, df_test
