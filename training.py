from typing import Optional

from data_engineering import data_engineering
from utils import get_possible_feature_eng


def select_feature_engineering(feat_eng: list[str], drop_list: Optional[list[bool]] = None) -> list:
    """
    This function selects pipeline-ready feature engineering transformations
    for the model training. The drop_list should specify a list of booleans if the 
    original variable should be droped after the feature engineering. The feat_eng should
    specify a list of strings, which feature engineering to select.
    """

    # get all possible transformations
    transf = get_possible_feature_eng(drop_list)

    # select all transformations from list
    selected_transf = {key: transf[key] for key in feat_eng}
    
    # return transformations as list of tuples
    return list(selected_transf.items())