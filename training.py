from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from utils import get_possible_feature_eng, load_train_data
from modeling.estimators import get_estimator


# TODO: take estimator, feat_eng and drop list out of the function training_estimator and implement in modeling or something


def select_feature_engineering(
    feat_eng: list[str], drop_list: Optional[list[bool]] = None
) -> list:
    """
    This function selects pipeline-ready feature engineering transformations
    for the model training. The drop_list should specify a list of booleans if the
    original variable should be droped after the feature engineering. The feat_eng should
    specify a list of strings, which feature engineering to select.
    """
    # get all possible transformations
    transf = get_possible_feature_eng(drop_list)

    # select all transformations from list
    selected_transf = {key: transf[key] for key in feat_eng}

    # return transformations as list of tuples
    return list(selected_transf.items())


def get_training_pipeline(
    estimator,
    feat_eng: Optional[list[str]] = None,
    drop_list: Optional[list[bool]] = None,
) -> Pipeline:
    """
    This functions creates a sklearn pipeline (incl. data engineering, model)
    that can be used for training or tuning of models.
    """
    # get feature engineering functions
    if feat_eng is not None:
        engineering = select_feature_engineering(feat_eng, drop_list)
    else:
        engineering = []

    # concat engineering and estimator and make sklearn pipeline
    pipe_steps = engineering + [("model", estimator)]
    pipe = Pipeline(pipe_steps)

    return pipe


def training_cv(model_class: str) -> None:
    """
    This function trains an estimator on a 5-fold cv of the train data set
    and prints the train and test accuracy on every fold.
    """
    # define the input varibales
    X, y = load_train_data()
    estimator, feat_eng, drop_list, grid = get_estimator(model_class)
    cv = KFold(shuffle=True)

    # load pipe
    pipe = get_training_pipeline(estimator, feat_eng, drop_list)
    if grid is not None:
        pipe.set_params(**grid)

    # perform cross validation on training data
    cv_results = cross_validate(
        estimator=pipe,
        X=X,
        y=y,
        cv=cv,
        n_jobs=4,
        scoring="accuracy",
        return_train_score=True,
        return_estimator=True,
    )

    # print results to track output
    for i in range(5):
        print(
            f"Fold {i}: training accuracy {cv_results['train_score'][i]:.3f}, testing accuracy {cv_results['test_score'][i]:.3f}"
        )
    print(f"----------- Mean train accuracy: {np.mean(cv_results['train_score']):.3f} -----------")
    print(f"----------- Mean test accuracy:  {np.mean(cv_results['test_score']):.3f} -----------")


def training_estimator(model_class: str) -> Pipeline:
    """
    This function trains an estimator on the whole data sets and returns the model.
    """
    # define the input varibales
    X, y = load_train_data()
    estimator, feat_eng, drop_list, grid = get_estimator(model_class)

    # get pipe and fit it on the train data 
    pipe = get_training_pipeline(estimator, feat_eng, drop_list)
    if grid is not None:
        pipe.set_params(**grid)
    pipe.fit(
        X=X,
        y=y,
    )

    print(f"----------- Train accuracy: {pipe.score(X, y):.3f} -----------")

    return pipe
