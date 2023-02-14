from sklearn.linear_model import LogisticRegression

# TODO: think about options to take out feat_eng, drop_list, grid so that this function is unchanged for every run
#       e.g. create an excel that is read as input params and which is part of gitignore


def get_estimator(model_class: str) -> tuple:
    """
    This function returns the right estimator according to the specified 
    model class and returns the feature engineering selection inputs and 
    hyperparameter grid.
    """
    # change the values of the following three variables
    feat_eng = ["scaling"]
    drop_list = None
    grid = {"model__penalty": "none"}

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
