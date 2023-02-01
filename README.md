# Forest Cover Type Prediction - Map541-2022-2023

Link: https://www.kaggle.com/competitions/map541-2022-2023/overview

Authors: Arianna Morè, João Melo, Maria Stoelben

The goal is to predict the forest cover type (a categorical variable) from cartographic variables only (no remotely sensed data). The actual forest cover type for a given observation (30 x 30 meter cell) was determined from US Forest Service (USFS) Region 2 Resource Information System (RIS) data. Independent variables were derived from data originally obtained from US Geological Survey (USGS) and USFS data. Data is in raw form (not scaled) and contains binary (0 or 1) columns of data for qualitative independent variables (wilderness areas and soil types). 
This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.

## The Data
The data consists of observations of 30m x 30m patches in four wilderness ares in the Roosevelt National Forest of northern Colorado. Twelve independent variables were derived using data from the US Geological Survey and US Forest Services:
* Elevation, quantitative (meters): Elevation in meters
* Aspect, quantitative (azimuth): Aspect in degrees azimuth
* Slope, quantitative (degrees): Slope in degrees
* Horizontal_Distance_To_Hydrology , quantitative (meters): Horz Dist to nearest surface water features
* Vertical_Distance_To_Hydrology , quantitative (meters): Vert Dist to nearest surface water features
* Horizontal_Distance_To_Roadways , quantitative (meters ): Horz Dist to nearest roadway
* Hillshade_9am , quantitative (0 to 255 index): Hillshade index at 9am, summer solstice
* Hillshade_Noon, quantitative (0 to 255 index): Hillshade index at noon, summer soltice
* Hillshade_3pm, quantitative (0 to 255 index): Hillshade index at 3pm, summer solstice
* Horizontal_Distance_To_Fire_Points, quantitative (meters): Horz Dist to nearest wildfire ignition points
* Wilderness_Area (4 binary columns), qualitative (0 (absence) or 1 (presence)): Wilderness area designation
* Soil_Type (40 binary columns), qualitative ( 0 (absence) or 1 (presence)): Soil Type designation

From these, we want to predict:
* Cover_Type (integers 1 to 7):
    1.  Spruce/Fir
    2.  Lodgepole Pine
    3.  Ponderosa Pine
    4.  Cottonwood/Willow
    5.  Aspen
    6.  Douglas-fir
    7.  Krummholz

Kaggle provides a labeled training data set of 15,120 observations, and an unlabeled test data set of 581,012 observations to be used for the submission.

* training set of 15,120 observations
* test set of 581,012 observations
* sample submission

