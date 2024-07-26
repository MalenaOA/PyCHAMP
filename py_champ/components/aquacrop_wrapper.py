
import os
from os import chdir, getcwd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
from aquacrop.utils import prepare_weather, get_filepath
import seaborn as sns
import numpy as np
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings('ignore')

def run_aquacrop(csv_file_path):

    os.chdir('/Users/michellenguyen/Downloads/PyCHAMP/examples/Heterogeneity/calibration_example 2')  # change working directory

#         wd = getcwd()

#         from src.soils import *
#         from src.calibration_old import *

#         # Input files
#         with open(wd + '/data/input_dict.pickle', 'rb') as input_data:
#             input_dict = pickle.load(input_data)

#         planting_date = pd.read_csv(wd + '/data/CropPlantingDate_GMD4_WNdlovu_072423.csv')
#         bias_correction = pd.read_csv(wd + '/data/CropBiasCorrectionParams_GMD4_Wndlovu_062424.csv')

#         # Use the CSV file produced by PyCHAMP
#         defaults = pd.read_csv(csv_file_path)  # default model params

#         irrig_crop = '1'  # code for irrigated corn
#         irrig_crop_dict = {k: v for (k, v) in input_dict.items() if irrig_crop in k}

#         calibration = list(irrig_crop_dict.items())
#         gridmet = pd.concat([sublist[1][0] for sublist in calibration])
#         et = pd.concat([sublist[1][1] for sublist in calibration])
#         soil_irrig = pd.concat(SoilCompart([sublist[1][2] for sublist in calibration]))

#         planting_date = planting_date[planting_date["Crop"] == 'Maize']
#         planting_date['pdate'] = pd.to_datetime(planting_date['pdate'], format='%Y-%m-%d')
#         planting_date['har'] = pd.to_datetime(planting_date['har'], format='%Y-%m-%d')
#         planting_date['late_har'] = pd.to_datetime(planting_date['late_har'], format='%Y-%m-%d')

#         planting_date['pdate'] = planting_date['pdate'].dt.strftime('%Y/%m/%d')
#         planting_date['har'] = planting_date['har'].dt.strftime('%Y/%m/%d')
#         planting_date['late_har'] = planting_date['late_har'].dt.strftime('%Y/%m/%d')

#         planting_date['canopy'] = 0.96 / 0.012494

#         gridmet_2009 = gridmet[(gridmet['Year'] == 2009) & (gridmet['crop_mn_codeyear'] == '1_Cheyenne')]

#         example_df = RunModelBiasCorrected(defaults, planting_date, gridmet_2009, soil_irrig, bias_correction, 4, 11)
#         print(example_df)

#         absolute_path = '/Users/michellenguyen/Downloads/example_df_full.csv'
#         example_df.to_csv(absolute_path, index=False)

#         return example_df
    