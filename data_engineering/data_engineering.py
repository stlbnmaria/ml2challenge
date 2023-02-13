import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# TODO: drop used columns so that in modeling there are no linear combinations etc.


def euclidean_dist(X: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the eucledian distance to nearest surface water features
    given a horizontal and vertical distance.
    """
    X = X.copy()  # modify a copy of X

    # Compute the Euclidean distance to Hydrology
    hv_distances_labels = [
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
    ]
    hv_dist_hydro_arr = X[hv_distances_labels].values
    euc_distance_to_hydro = np.linalg.norm(hv_dist_hydro_arr, axis=1)
    X.insert(3, "Distance_To_Hydrology", np.around(euc_distance_to_hydro, decimals=4))
    X.Distance_To_Hydrology = X.Distance_To_Hydrology.map(
        lambda x: 0 if np.isinf(x) else x
    )  # remove infinite value if any
    return X


def linear_dist(X: pd.DataFrame) -> pd.DataFrame:
    """
    This functions calculates linear combinations of distances to hydrology, 
    fire points and roadways, inlcuding:
        - hydrology + fire points
        - |hydrology - fire points|
        - hydrology + roadways
        - |hydrology - roadways|
        - fire + roadways
        - |fire - roadways|
    """
    X = X.copy()  # modify a copy of X

    cols = [
        X["Horizontal_Distance_To_Hydrology"],
        X["Horizontal_Distance_To_Fire_Points"],
        X["Horizontal_Distance_To_Roadways"],
    ]

    # Linear Combination of distance
    X["Hyd_p_Fire"] = cols[0] + cols[1]
    X["Hyd_m_Fire"] = abs(cols[0] - cols[1])
    X["Hyd_p_Road"] = cols[0] + cols[2]
    X["Hyd_m_Road"] = abs(cols[0] - cols[2])
    X["Fire_p_Road"] = cols[1] + cols[2]
    X["Fire_m_Road"] = abs(cols[1] - cols[2])
    return X


def mean_hillshade(X: pd.DataFrame) -> pd.DataFrame:
    """
    This functions calculates a mean hillshade for every observation of X, 
    i.e., it averages the three hillshades at 9 am, noon and 3pm to create 
    one hillshade value for one tree. 
    """
    X = X.copy()  # modify a copy of X

    # Mean distance to Hillshade
    X["Mean_Hillshade"] = (X.Hillshade_9am + X.Hillshade_Noon + X.Hillshade_3pm) / 3
    return X


def morning_hillshade(X: pd.DataFrame) -> pd.DataFrame:
    """
    This functions calculates a mean morning hillshade for every observation of X, 
    i.e., it averages the two hillshades at 9 am and noon to create 
    one hillshade value for the morning for one tree. 
    """
    X = X.copy()  # modify a copy of X

    # Mean distance to Hillshade
    X["Morning_Hillshade"] = (X.Hillshade_9am + X.Hillshade_Noon) / 2
    return X


def mean_amenties(X: pd.DataFrame) -> pd.DataFrame:
    """
    This functions calculates the average distance to hydrology, 
    fire points and roadways.
    """
    X = X.copy()  # modify a copy of X

    # Mean distance to Amenities
    X["Mean_Amenities"] = (
        X.Horizontal_Distance_To_Fire_Points
        + X.Horizontal_Distance_To_Hydrology
        + X.Horizontal_Distance_To_Roadways
    ) / 3
    return X


def aspect_dir(X: pd.DataFrame) -> pd.DataFrame:
    """
    This functions converts the degree values of the variable aspect
    into four main directions.
        - North: 0-45 degrees and 315-360 degrees
        - East: 45-135 degrees
        - South: 135-225 degrees
        - West: 225-325 degrees
    """
    X = X.copy()  # modify a copy of X

    asp = X["Aspect"]

    # Splitting Aspect in four directions
    X["Aspect_N"] = np.where(((asp >= 0) & (asp < 45)) | ((asp >= 315) & (asp <= 360)), 1, 0)
    X["Aspect_E"] = np.where((asp >= 45) & (asp < 135), 1, 0)
    X["Aspect_S"] = np.where((asp >= 135) & (asp < 225), 1, 0)
    X["Aspect_W"] = np.where((asp >= 225) & (asp < 315), 1, 0)
    return X


def climatic_zone(X: pd.DataFrame) -> pd.DataFrame:
    """
    This functions aggregates the second digit of the USFS Ecological Landtype Units which 
    encodes the following climatic zones: 
        1: lower montane dry
        2: lower montane
        3: montane dry
        4: montane
        5: montane dry and montane
        6: montane and subalpine
        7: subalpine
        8: alpine
    """
    X = X.copy()  # modify a copy of X

    # Climatic Zones
    X["Clim2"] = X.loc[:, X.columns.str.contains("^Soil_Type[2-6]$")].max(axis=1)
    X["Clim3"] = X.loc[:, X.columns.str.contains("^Soil_Type[78]$")].max(axis=1)
    X["Clim4"] = X.loc[:, X.columns.str.contains("^Soil_Type[1][0-3]$|Soil_Type9")].max(
        axis=1
    )
    X["Clim5"] = X.loc[:, X.columns.str.contains("^Soil_Type[1][45]$")].max(axis=1)
    X["Clim6"] = X.loc[:, X.columns.str.contains("^Soil_Type[1][678]$")].max(axis=1)
    X["Clim7"] = X.loc[:, X.columns.str.contains("^Soil_Type19$|^Soil_Type[2][0-9]$|^Soil_Type[3][0-4]$")].max(axis=1)
    X["Clim8"] = X.loc[:, X.columns.str.contains("^Soil_Type[3][56789]$|Soil_Type40")].max(axis=1)
    return X


def geologic_zone(X: pd.DataFrame) -> pd.DataFrame:
    """
    This functions aggregates the second digit of the USFS Ecological Landtype Units which 
    encodes the following geologic zones: 
        1: alluvium
        2: glacial
        3: shale
        4: sandstone
        5: mixed sedimentary
        6: unspecified in the USFS ELU Survey
        7: igneous and metamorphic
        8: volcanic
    """
    X = X.copy()  # modify a copy of X

    # geologic zones
    X["Geo1"] = X.loc[:, X.columns.str.contains("^Soil_Type[1][45679]$|^Soil_Type[2][01]$")].max(axis=1)
    X["Geo2"] = X.loc[:, X.columns.str.contains("^Soil_Type[9]$|^Soil_Type[2][23]$")].max(axis=1)
    X["Geo5"] = X.loc[:, X.columns.str.contains("^Soil_Type[7-8]$")].max(axis=1)
    X["Geo7"] = X.loc[:, X.columns.str.contains("^Soil_Type[1-6]$|^Soil_Type[1][01238]$|^Soil_Type[3-4]\d$|^Soil_Type[2][4-9]$")].max(axis=1)
    return X


def soil_type(X: pd.DataFrame) -> pd.DataFrame:
    """
    This functions aggregates the different soil types in the USFS Ecological 
    Landtype Units description which encodes the following: 
        1: stony
        2: rubly
        3: other
    """
    X = X.copy()  # modify a copy of X

    # Soil Type
    X["Soil_Stony"] = X.loc[:, X.columns.str.contains("^Soil_Type[1269]$|^Soil_Type[1][28]$|^Soil_Type[2][456789]$|^Soil_Type[3][012346789]$")].max(axis=1)
    X["Soil_Rubly"] = X.loc[:, X.columns.str.contains("^Soil_Type[345]$|^Soil_Type[1][0123]$")].max(axis=1)
    X["Soil_Other"] = X.loc[:, X.columns.str.contains("^Soil_Type[78]$|^Soil_Type[1][45679]$|^Soil_Type[2][0123]$")].max(axis=1)
    return X


def scaling(X: pd.DataFrame) -> pd.DataFrame:
    """
    This function does a standard scaling on all numerical columns in the data set 
    and returns the whole df.
    """
    X = X.copy()  # modify a copy of X

    # scale numerical values
    X.drop(columns="Id", inplace=True)
    X_col = X.columns
    X_possible_col = [
        "Elevation",
        "Aspect",
        "Distance_To_Hydrology",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
        "Hyd_p_Fire",
        "Hyd_m_Fire",
        "Hyd_p_Road",
        "Hyd_m_Road",
        "Fire_p_Road",
        "Fire_m_Road",
        "Mean_Hillshade",
        "Morning_Hillshade",
        "Mean_Amenities",
    ]
    X_num_columns = list(set(X_col) & set(X_possible_col))
    X_num = X[X_num_columns]
    X_cat = X.drop(columns=X_num_columns)
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num.to_numpy())
    X_num_scaled = pd.DataFrame(X_num_scaled, columns=X_num_columns)
    X_rs = X_num_scaled.join(X_cat)
    return X_rs
