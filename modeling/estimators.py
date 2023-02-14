from sklearn.linear_model import LogisticRegression


def get_estimator(model_class: str) -> tuple:
    # change the values of the following three variables
    feat_eng = ["scaling"]
    drop_list = None
    grid = {"model__penalty": "none"}

    # get estimator based on specified model class
    if model_class == "LogReg":
        estimator = LogisticRegression()
    else: 
        print("Specified estimator could not be found in possible list. Used default LogisticRegression() instead.")
        estimator = LogisticRegression()
        feat_eng = ["scaling"]
        drop_list = None
        grid = None

    return estimator, feat_eng, drop_list, grid
