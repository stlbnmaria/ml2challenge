
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

    

def euclidean_dist(X: pd.DataFrame):
    # Compute the Euclidean distance to Hydrology
    hv_distances_labels = ['Horizontal_Distance_To_Hydrology',
                       'Vertical_Distance_To_Hydrology']
    hv_dist_hydro_arr = X[hv_distances_labels].values
    euc_distance_to_hydro = np.linalg.norm(hv_dist_hydro_arr, axis=1)
    X.insert(3,'Distance_To_Hydrology',
                np.around(euc_distance_to_hydro, decimals=4))
    X.Distance_To_Hydrology=X.Distance_To_Hydrology.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any
    return X

def linear_dist(X: pd.DataFrame):
    #Linear Combination of distance
    X['Hyd_p_Fire'] = X['Horizontal_Distance_To_Hydrology'] + X['Horizontal_Distance_To_Fire_Points']
    X['Hyd_m_Fire'] = abs(X['Horizontal_Distance_To_Hydrology'] - X['Horizontal_Distance_To_Fire_Points'])
    X['Hyd_p_Road'] = X['Horizontal_Distance_To_Hydrology'] + X['Horizontal_Distance_To_Roadways']
    X['Hyd_m_Road'] = abs(X['Horizontal_Distance_To_Hydrology'] - X['Horizontal_Distance_To_Roadways'])
    X['Fire_p_Road'] = X['Horizontal_Distance_To_Fire_Points'] + X['Horizontal_Distance_To_Roadways']
    X['Fire_m_Road'] = abs(X['Horizontal_Distance_To_Fire_Points'] - X['Horizontal_Distance_To_Roadways'])
    return X

def mean_hillshade(X: pd.DataFrame):
    #Mean distance to Hillshade 
    X['Mean_Hillshade'] = (X.Hillshade_9am + X.Hillshade_Noon + X.Hillshade_3pm) / 3 
    return X

def morning_hillshade(X: pd.DataFrame):
    #Mean distance to Hillshade 
    X['Morning_Hillshade'] = (X.Hillshade_9am + X.Hillshade_Noon) / 2 
    return X


def mean_amenties(X: pd.DataFrame):
    #Mean distance to Amenities 
    X['Mean_Amenities'] = (X.Horizontal_Distance_To_Fire_Points + X.Horizontal_Distance_To_Hydrology + X.Horizontal_Distance_To_Roadways) / 3
    return X

def aspect_dir(X: pd.DataFrame):
    #Splitting Aspect in four directions
    X['Aspect_N'] =  np.where(((X['Aspect']>=0) & (X['Aspect']<45))|((X['Aspect']>=315) & (X['Aspect']<=360)), 1 ,0)
    X['Aspect_E'] = np.where((X['Aspect']>=45) & (X['Aspect']<135), 1 ,0)
    X['Aspect_S'] = np.where((X['Aspect']>=135) & (X['Aspect']<225), 1 ,0)
    X['Aspect_W'] = np.where((X['Aspect']>=225) & (X['Aspect']<315), 1 ,0)
    return X


def climatic_zone(X: pd.DataFrame):
    #Climatic Zones
    X['Clim2'] =X.loc[:,X.columns.str.contains("^Soil_Type[2-6]$")].max(axis=1)
    X['Clim3'] =X.loc[:,X.columns.str.contains("^Soil_Type[78]$")].max(axis=1)
    X['Clim4'] =X.loc[:,X.columns.str.contains("^Soil_Type[1][0-3]$|Soil_Type9")].max(axis=1)
    X['Clim5'] =X.loc[:,X.columns.str.contains("^Soil_Type[1][45]$")].max(axis=1)
    X['Clim6'] =X.loc[:,X.columns.str.contains("^Soil_Type[1][678]$")].max(axis=1)
    X['Clim7'] =X.loc[:,X.columns.str.contains("^Soil_Type19$|^Soil_Type[2][0-9]$|^Soil_Type[3][0-4]$")].max(axis=1)
    X['Clim8'] =X.loc[:,X.columns.str.contains("^Soil_Type[3][56789]$|Soil_Type40")].max(axis=1)
    return X


def geologic_zone(X: pd.DataFrame):
    #geologic zones
    X['Geo1'] = X.loc[:,X.columns.str.contains("^Soil_Type[1][45679]$|^Soil_Type[2][01]$")].max(axis=1)
    X['Geo2'] =X.loc[:,X.columns.str.contains("^Soil_Type[9]$|^Soil_Type[2][23]$")].max(axis=1)
    X['Geo5'] =X.loc[:,X.columns.str.contains("^Soil_Type[7-8]$")].max(axis=1)
    X['Geo7'] =X.loc[:,X.columns.str.contains("^Soil_Type[1-6]$|^Soil_Type[1][01238]$|^Soil_Type[3-4]\d$|^Soil_Type[2][4-9]$")].max(axis=1)
    return X


def soil_type(X: pd.DataFrame):
# Soil Type
    X['Soil_Stony']= X.loc[:,X.columns.str.contains("^Soil_Type[1269]$|^Soil_Type[1][28]$|^Soil_Type[2][456789]$|^Soil_Type[3][012346789]$")].max(axis=1)
    X['Soil_Rubly']= X.loc[:,X.columns.str.contains("^Soil_Type[345]$|^Soil_Type[1][0123]$")].max(axis=1)
    X['Soil_Other']= X.loc[:,X.columns.str.contains("^Soil_Type[78]$|^Soil_Type[1][45679]$|^Soil_Type[2][0123]$")].max(axis=1)
    return X


def scaling(X: pd.DataFrame):
    # scale numerical values
    X.drop(columns='Id', inplace=True)
    X_col = X.columns
    X_possible_col = ['Elevation', 'Aspect', 'Distance_To_Hydrology', 'Slope',
                    'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
                    'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 
                    'Hyd_p_Fire', 'Hyd_m_Fire', 'Hyd_p_Road', 'Hyd_m_Road',
                    'Fire_p_Road', 'Fire_m_Road', 'Mean_Hillshade', 'Morning_Hillshade', 'Mean_Amenities'
                    ]
    X_num_columns = list(set(X_col) & set(X_possible_col))
    X_num = X[X_num_columns]
    X_cat = X.drop(columns=X_num_columns)
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num.to_numpy())
    X_num_scaled = pd.DataFrame(X_num_scaled, columns=X_num_columns)
    X_rs = X_num_scaled.join(X_cat)
    return X_rs

