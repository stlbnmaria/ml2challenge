import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from utils import load_test_data
from training import training_estimator

# TODO: write function to plot confusion matrix to understand predictions


def predict_on_test(classifier: Pipeline) -> tuple[np.array]:
    """
    This functions predicts class of every observation on the test set
    given a trained classifier. It returns the test ids and predictions.
    """
    # load test data and create df to predict on and
    X_test = load_test_data()
    test_id = X_test["Id"].to_numpy()

    # calcualte predictions
    preds = classifier.predict(X_test)

    # check that the prediction vetor has the right size
    assert preds.shape == (len(test_id),)

    return test_id, preds


def create_submission(classifier: Pipeline, sub_name: str, path: str = "./submissions"):
    """
    This function takes as input a list of k classifiers from a k-fold cv training
    and saves the predictions in submission format as csv.
    """
    test_id, preds = predict_on_test(classifier)
    submission = pd.DataFrame({"Id": test_id, "Cover_Type": preds})
    timestamp = datetime.today().strftime("%Y%m%d_%H%M")
    sub_name = timestamp + "_" + sub_name + ".csv"

    if not os.path.exists(path):
        os.makedirs(path)

    submission.to_csv(os.path.join(path, sub_name), index=False)
    print("----------- Submission saved successfully -----------")


def run_train_submission() -> None:
    """
    This function runs the training of a classifier on the whole train data set,
    does predictions on the test set and saves a csv ready for submission.
    """
    cv_classifiers = training_estimator()
    create_submission(cv_classifiers, "test")
