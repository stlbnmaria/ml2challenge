import argparse
from pathlib import Path

from training import training_estimator

###############################################################################

parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
parser.add_argument('--model', type = str, default = r"XGB", help = 'Type of model to run training')
parser.add_argument('--goal', type = str, default = r"test", help = 'Goal of the run (test/submission/tuning)')
parser.add_argument('--subname', type = str, default = r"test", help = 'Name of the submission')
args = parser.parse_args()  

###############################################################################

if __name__ == "__main__":

    input_args = vars(args)
    
    training_estimator()
