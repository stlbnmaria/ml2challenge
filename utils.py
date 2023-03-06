import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, LabelEncoder

from data_engineering import data_engineering


def get_possible_feature_eng(drop_list: Optional[list] = None) -> dict:
    """
    This function returns a dict with all possible data engineering functions
    that can be used in training which are ready to be used in a pipeline
    as they have been transformed by FunctionTransformer from sklearn.
    The drop_list should specify a list of booleans if the original variable should
    be droped after the feature engineering. If it is None, no variables will be droped.
    """
    if drop_list is None:
        drop_list = [False] * 9

    transformations = {
        "euclidean_dist": FunctionTransformer(
            data_engineering.euclidean_dist, kw_args={"drop_original": drop_list[0]}
        ),
        "linear_dist": FunctionTransformer(
            data_engineering.linear_dist, kw_args={"drop_original": drop_list[1]}
        ),
        "mean_hillshade": FunctionTransformer(
            data_engineering.mean_hillshade, kw_args={"drop_original": drop_list[2]}
        ),
        "morning_hillshade": FunctionTransformer(
            data_engineering.morning_hillshade, kw_args={"drop_original": drop_list[3]}
        ),
        "mean_amenties": FunctionTransformer(
            data_engineering.mean_amenties, kw_args={"drop_original": drop_list[4]}
        ),
        "aspect_dir": FunctionTransformer(
            data_engineering.aspect_dir, kw_args={"drop_original": drop_list[5]}
        ),
        "climatic_zone": FunctionTransformer(
            data_engineering.climatic_zone, kw_args={"drop_original": drop_list[6]}
        ),
        "geologic_zone": FunctionTransformer(
            data_engineering.geologic_zone, kw_args={"drop_original": drop_list[7]}
        ),
        "soil_type": FunctionTransformer(
            data_engineering.soil_type, kw_args={"drop_original": drop_list[8]}
        ),
        "scaling": FunctionTransformer(data_engineering.scaling),
    }
    return transformations


def load_train_data(
    path: str = "./data",
) -> tuple[pd.DataFrame, np.array, LabelEncoder]:
    file = "train.csv"
    df = pd.read_csv(os.path.join(path, file))
    X = df.drop(columns="Cover_Type")
    y = np.array(df.Cover_Type)

    le = LabelEncoder()
    y = le.fit_transform(y)

    # perform checks to assure size matches the expectations
    assert X.shape == (15_120, 55)
    assert y.shape == (15_120,)

    return X, y, le


def load_test_data(path: str = "./data") -> pd.DataFrame:
    file = "test-full.csv"
    df = pd.read_csv(os.path.join(path, file))

    # perform checks to assure size matches the expectations
    assert df.shape == (581_012, 55)

    return df
