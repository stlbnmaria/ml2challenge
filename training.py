from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from utils import get_possible_feature_eng, load_train_data


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


def training_estimator():
    # define the input varibales
    estimator = LogisticRegression()
    feat_eng = ["scaling"]
    drop_list = None

    # load pipe, train data and initialise cross validation split
    pipe = get_training_pipeline(estimator, feat_eng, drop_list)
    X, y = load_train_data()
    cv = KFold(shuffle=True)

    # perform cross validation on training data
    cv_results = cross_validate(
        estimator=pipe,
        X=X,
        y=y,
        cv=cv,
        scoring="accuracy",
        return_train_score=True,
        return_estimator=True,
    )

    # print results to track output
    for i in range(5):
        print(
            f"Fold {i}: training accuracy {cv_results['train_score'][i]:.3f}, testing accuracy {cv_results['test_score'][i]:.3f}"
        )
