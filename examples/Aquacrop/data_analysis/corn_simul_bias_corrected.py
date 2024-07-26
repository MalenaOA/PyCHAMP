import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
from aquacrop.utils import prepare_weather, get_filepath
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement


import pandas as pd
import matplotlib
import statsmodels
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import time
import math
import pickle 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sklearn.metrics as metrics
#from pyswarm import pso
import os
from os import chdir, getcwd
import statistics as stats
import scipy
from scipy import stats
#import hydroeval as he
#import pyswarms as ps
#from pyswarm import pso
import random
from ordered_set import OrderedSet
import warnings
warnings.filterwarnings('ignore')


os.chdir('/Users/michellenguyen/Downloads/calibration_example 2') # change working directory

wd=getcwd()

from src.soils import *
from src.calibration_old import *


# input files
with open(wd + '/data/input_dict.pickle', 'rb') as input_data: 
    input_dict = pickle.load(input_data) 



planting_date = pd.read_csv(wd + '/data/CropPlantingDate_GMD4_WNdlovu_072423.csv')
bias_correction = pd.read_csv(wd + '/data/CropBiasCorrectionParams_GMD4_Wndlovu_062424.csv')



defaults = pd.read_csv(wd + '/data/CropDefaultParams_GMD4_Wndlovu_062424.csv') # default model params
defaults = defaults[defaults["Crop"] == 'Maize']


irrig_crop = '1' # code for irrigated corn
irrig_crop_dict = {k:v for (k,v) in input_dict.items() if irrig_crop in k}

calibration = list(irrig_crop_dict.items())
#

# getting the different datasets
gridmet = pd.concat([sublist[1][0] for sublist in calibration]) # data stored in nested list with 0 having the 
et = pd.concat([sublist[1][1] for sublist in calibration])
soil_irrig = pd.concat(SoilCompart([sublist[1][2] for sublist in calibration]))



planting_date = planting_date[planting_date["Crop"] == 'Maize']
planting_date['pdate'] = pd.to_datetime(planting_date['pdate'], format='%Y-%m-%d')
planting_date['har'] = pd.to_datetime(planting_date['har'], format='%Y-%m-%d')
planting_date['late_har'] = pd.to_datetime(planting_date['late_har'], format='%Y-%m-%d')

# Convert the 'date' column back to a new column in YMD format
planting_date['pdate'] = planting_date['pdate'].dt.strftime('%Y/%m/%d')
planting_date['har'] = planting_date['har'].dt.strftime('%Y/%m/%d')
planting_date['late_har'] = planting_date['late_har'].dt.strftime('%Y/%m/%d')

# add the ccx to cgc ratio for corn
planting_date['canopy'] = 0.96/0.012494 # change for different crops


# Run one unique id for one year

# run model for Cheyenne County (unique id) for one year
gridmet_2009 = gridmet[(gridmet['Year'] == 2009) & (gridmet['crop_mn_codeyear'] == '1_Cheyenne')]

example_df = RunModelBiasCorrected(defaults, planting_date, gridmet_2009, soil_irrig,  bias_correction, 4, 11)
print(example_df)

absolute_path = '/Users/michellenguyen/Downloads/example_df_full.csv'

# Save DataFrame to CSV
example_df.to_csv(absolute_path, index=False)