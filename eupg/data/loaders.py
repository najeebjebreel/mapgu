"""Dataset loaders for Adult, Credit, Heart, and CIFAR-10."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from torchvision import datasets as tv_datasets
from torchvision import transforms

from eupg.config import DATA_DIR, ADULT_NUM_COLS, ADULT_CAT_COLS, CIFAR_MEAN, CIFAR_STD
from eupg.data.preprocessors import ColumnsSelector, CategoricalImputer, CategoricalEncoder


def _adult_income_to_int(x: object) -> int:
    s = str(x).strip()
    return 0 if s in ["<=50K", "<=50K."] else 1


class DatasetLoader:
    @staticmethod
    def load_adult(return_raw: bool = False):
        columns = [
            "age", "workClass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
        ]

        train_path = os.path.join(DATA_DIR, "adult", "adult.data")
        test_path  = os.path.join(DATA_DIR, "adult", "adult.test")

        train_data = pd.read_csv(
            train_path, names=columns, sep=r" *, *", engine="python", na_values="?"
        )
        test_data = pd.read_csv(
            test_path, names=columns, sep=r" *, *", skiprows=1, engine="python", na_values="?"
        )

        for df in (train_data, test_data):
            df.drop(["fnlwgt", "education"], axis=1, inplace=True)
            df.dropna(subset=ADULT_NUM_COLS, inplace=True)
            df.reset_index(drop=True, inplace=True)

        train_raw = train_data.copy()
        test_raw  = test_data.copy()

        train_raw["income"] = train_raw["income"].apply(_adult_income_to_int)
        test_raw["income"]  = test_raw["income"].apply(_adult_income_to_int)

        X_train_df = train_raw.drop("income", axis=1)
        X_test_df  = test_raw.drop("income", axis=1)

        num_pipeline = Pipeline(
            steps=[
                ("num_attr_selector", ColumnsSelector(type="int")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("cat_attr_selector", ColumnsSelector(type="object")),
                ("cat_imputer", CategoricalImputer(
                    columns=["workClass", "occupation", "native-country"]
                )),
                ("encoder", CategoricalEncoder(train_df=None, test_df=None, dropFirst=True)),
            ]
        )

        full_pipeline = FeatureUnion([("num_pipe", num_pipeline), ("cat_pipeline", cat_pipeline)])

        X_train = full_pipeline.fit_transform(X_train_df)
        y_train = train_raw["income"].values

        X_test = full_pipeline.transform(X_test_df)
        y_test = test_raw["income"].values

        if return_raw:
            return (
                X_train, y_train,
                X_test,  y_test,
                full_pipeline,
                X_train_df,
                X_test_df,
            )
        return X_train, y_train, X_test, y_test, full_pipeline

    @staticmethod
    def load_credit(seed: int = 42):
        path = os.path.join(DATA_DIR, "credit", "cs-training.csv")
        df_train = pd.read_csv(path)
        if "Unnamed: 0" in df_train.columns:
            df_train.drop(columns=["Unnamed: 0"], inplace=True)

        df_train.dropna(inplace=True)
        y = df_train["SeriousDlqin2yrs"].values
        X = df_train.drop(["SeriousDlqin2yrs"], axis=1).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, y_train, X_test, y_test, scaler

    @staticmethod
    def load_heart(seed: int = 42):
        path = os.path.join(DATA_DIR, "heart", "cardio_train.csv")
        df = pd.read_csv(path, sep=";").dropna().reset_index(drop=True)

        if "id" in df.columns:
            df = df.drop(columns=["id"])

        df["age_years"] = df["age"] / 365.25
        df = df.drop(columns=["age"])

        y = df["cardio"].astype(int).values
        X_df = df.drop(columns=["cardio"])

        cont_cols = ["age_years", "height", "weight", "ap_hi", "ap_lo"]
        cat_cols  = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

        preprocessor = ColumnTransformer(
            [
                ("cont", StandardScaler(), cont_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ],
            remainder="drop",
        )

        X = preprocessor.fit_transform(X_df).astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        return X_train, y_train, X_test, y_test, preprocessor

    @staticmethod
    def load_cifar10(download: bool = True):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

        root = DATA_DIR
        trainset = tv_datasets.CIFAR10(root=root, train=True, download=download, transform=transform_train)
        testset = tv_datasets.CIFAR10(root=root, train=False, download=download, transform=transform_test)
        return trainset, testset
