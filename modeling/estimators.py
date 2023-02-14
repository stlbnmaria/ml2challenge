import os

from ast import literal_eval
import pandas as pd
from sklearn.linear_model import LogisticRegression


def get_modeling_inputs(path: str = "./modeling/") -> tuple:
    """
    This function loads the modeling inputs from a predefined excel input,
    inlc. the feature engineering selection inputs and
    hyperparameter grid.
    """
    modeling_inputs = pd.read_excel(os.path.join(path, "modeling_inputs.xlsx"))
    # extract specified data engineering
    feat_eng = modeling_inputs["feat_eng"][0]
    try:
        feat_eng = feat_eng.split(", ")
    except:
        feat_eng = None

    # extract drop list
    drop_list = modeling_inputs["drop_list"][0]
    try:
        drop_list = drop_list.split(", ")
        drop_list = [literal_eval(drop) for drop in drop_list]
    except:
        drop_list = None

    # extract hyperparameter grid
    grid = modeling_inputs["grid"][0]
    try:
        grid = literal_eval(grid)
    except:
        grid = None

    return feat_eng, drop_list, grid


def get_estimator(model_class: str) -> tuple:
    """
    This function returns the right estimator according to the specified
    model class and returns the feature engineering selection inputs and
    hyperparameter grid.
    """
    feat_eng, drop_list, grid = get_modeling_inputs()

    # get estimator based on specified model class
    if model_class == "LogReg":
        estimator = LogisticRegression()
    else:
        print(
            "Specified estimator could not be found in possible list. Used default LogisticRegression() instead."
        )
        estimator = LogisticRegression()
        feat_eng = ["scaling"]
        drop_list = None
        grid = None

    return estimator, feat_eng, drop_list, grid
