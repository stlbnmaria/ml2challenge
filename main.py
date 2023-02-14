import argparse

from training import training_cv
from inference import run_train_submission

###############################################################################

parser = argparse.ArgumentParser(description="Main Script to Run Training")
parser.add_argument(
    "--model", type=str, default=r"LogReg", help="Type of model to run training"
)
parser.add_argument(
    "--goal", type=str, default=r"test", help="Goal of the run (test/submission/tuning)"
)
parser.add_argument(
    "--subname", type=str, default=r"test", help="Name of the submission"
)
args = parser.parse_args()

###############################################################################

if __name__ == "__main__":

    input_args = vars(args)

    if input_args["goal"] == "test":
        training_cv(model_class=input_args["model"])
    elif input_args["goal"] == "submission":
        run_train_submission(
            model_class=input_args["model"], sub_name=input_args["subname"]
        )
