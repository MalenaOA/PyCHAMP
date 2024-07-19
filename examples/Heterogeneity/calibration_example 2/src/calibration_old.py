from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop import Crop
import pandas as pd
import numpy as np
import statistics as stats
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sklearn.metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
import hydroeval as he
import math
from pyswarm import pso


stata_palette = ['#1a476f', '#90353b', '#55752f', '#e37e00', '#6e8e84', 
           '#c10534', '#938dd2', '#cac27e', '#a0522d', '#7b92a8',
           '#2d6d66', '#9c8847', '#bfa19c', '#ffd200', '#d9e6eb']

et_train_palette = ['#2d6d66', '#9c8847', '#bfa19c']

et_test_palette = ['#a0522d', '#7b92a8', '#ffd200', '#d9e6eb']

et_full_palette = ['#2d6d66', '#9c8847', '#bfa19c', '#ffd200', '#d9e6eb']

uchicago_palette = ['#cac27e', '#a0522d', '#7b92a8',
           '#2d6d66', '#9c8847', '#bfa19c', '#ffd200', '#d9e6eb']



def CornHailImpacts(observed_values):
    """
    Function to identify years impacted by hail (western Kansas)

    Arguments:
        observed_values (DataFrame): dataframe with observed yield, irrigation values which includes the county and year

    Returns:

        hail (bool): bool identifying hail and non-hail entries
    """

    hail = ((observed_values['Year'] == 2006) & (observed_values['County'].isin(['Sheridan'])) |
            (observed_values['Year'] == 2009) & (observed_values['County'].isin(['Sherman', 'Thomas'])) |
            (observed_values['Year'] == 2010) & (observed_values['County'] == 'Thomas') |
            (observed_values['Year'] == 2011) & (observed_values['County'].isin(['Thomas', 'Sheridan'])) |
            (observed_values['Year'] == 2014) & (observed_values['County'].isin(['Sherman', 'Cheyenne'])) |
            (observed_values['Year'] == 2016) & (observed_values['County'].isin(['Sherman', 'Thomas'])) |
            (observed_values['Year'] == 2017) & (observed_values['County'] == 'Sheridan') |
            (observed_values['Year'] == 2018) & (observed_values['County'].isin(['Sherman', 'Thomas'])) |
            (observed_values['Year'] == 2019) & (observed_values['County'] == 'Wallace'))
    
    return hail




def FitnessCalc(x, *args):
    """
    Function to calculate the Particle Swarm Optimization algorithm objective function. Model simulations are compared to observed values.

    Arguments:
        x (list): list of calibration values used by the model to simulate crop-water productivity.
        args: Additional parameters passed to the function. Should include the following:
            - default_params (DataFrame): default parameters for the model that are specific to the study area
            - gdd (DataFrame): growing Degree Days data
            - calib (DataFrame): list with calibrated values from first calibration if 2-step calibration is used. Other generate an empty dataframe with variable, value and calib_val columns
            - plnt_date (DataFrame): planting date
            - gridmet_data (DataFrame): gridMET data
            - soil_data (DataFrame): soil data
            - iwc (str): initial water content ('FC': Field Capacity, 'WP': Wilting Point, 'SAT': Saturation
            - sa (list): sensitive paramaters.
            - observed_yield (Dataframe): observed crop yield
            - observed_irrig (DataFrame): observed irrigation data
            - observed_et (DataFrame): Observed evapotranspiration data
            - irrig_method (int): irrigation method
            - start_grwth_period (int): month corresponding to start of the growth period
            - end_grwth_period (int): month corresponding to end of the growth period
            - df_type (str): type of data frame (train or test)

    Returns:

        fitness (int): fitness score calculated using the weighted least squares objective function. Score can be for the train and test datasets.
    """
    
    default_params, gdd, calib, plnt_date, gridmet_data, soil_data, iwc, sa, observed_yield, observed_irrig, observed_et, irrig_method, start_grwth_period, end_grwth_period, df_type = args
        
    if df_type == 'train':
        model_comp_train = RunAllCounties(x, *args)[1][0]
        y_yield_train = model_comp_train[['YieldUSDA']].values
        yhat_yield_train = model_comp_train[['Calib Yield (t/ha)']].values

        y_irrig_train = model_comp_train[['irrig_depth']].values
        yhat_irrig_train = model_comp_train[['Calib Irrigation (mm)']].values

        std_yield_train = np.nanstd(y_yield_train - yhat_yield_train)
        std_irrig_train = np.nanstd(y_irrig_train - yhat_irrig_train)

        var_yield_train = std_yield_train**2
        var_irrig_train = std_irrig_train**2

        # Ensure variances are not zero to avoid division by zero
        var_yield_train = np.nan_to_num(var_yield_train, nan=1.0, posinf=1.0, neginf=1.0)
        var_irrig_train = np.nan_to_num(var_irrig_train, nan=1.0, posinf=1.0, neginf=1.0)

        # equally weighted objective function (yield and irrigation)
        if var_irrig_train == 0:  # expect var of 0 for rainfed simulation where irrig depths are 0
            fitness = np.nansum(((y_yield_train - yhat_yield_train) ** 2) * (1 / var_yield_train))
        else:
            fitness = (np.nansum(((y_yield_train - yhat_yield_train) ** 2) * (1 / var_yield_train)) 
                       + np.nansum(((y_irrig_train - yhat_irrig_train) ** 2) * (1 / var_irrig_train)))
 
        print(fitness)   
        return fitness 
        
        
    elif df_type == 'test':
        model_comp_test = RunAllCounties(x, *args)[2][0]
        model_comp_test = model_comp_test[model_comp_test.County != 'Sheridan']  # remove Sheridan LEMA from fitness score

        y_yield_test = model_comp_test[['YieldUSDA']].values
        yhat_yield_test = model_comp_test[['Calib Yield (t/ha)']].values

        y_irrig_test = model_comp_test[['irrig_depth']].values
        yhat_irrig_test = model_comp_test[['Calib Irrigation (mm)']].values

        # If y_et_test and yhat_et_test are relevant, they should be defined or included similarly to yield and irrigation
        y_et_test = model_comp_test[['ET']] if 'ET' in model_comp_test.columns else np.zeros_like(y_yield_test)
        yhat_et_test = model_comp_test[['Calib ET (mm)']] if 'Calib ET (mm)' in model_comp_test.columns else np.zeros_like(yhat_yield_test)

        std_yield_test = np.nanstd(y_yield_test - yhat_yield_test)
        std_irrig_test = np.nanstd(y_irrig_test - yhat_irrig_test)
        std_et_test = np.nanstd(y_et_test - yhat_et_test)

        var_yield_test = std_yield_test**2
        var_irrig_test = std_irrig_test**2
        var_et_test = std_et_test**2

        # Ensure variances are not zero to avoid division by zero
        var_yield_test = np.nan_to_num(var_yield_test, nan=1.0, posinf=1.0, neginf=1.0)
        var_irrig_test = np.nan_to_num(var_irrig_test, nan=1.0, posinf=1.0, neginf=1.0)
        var_et_test = np.nan_to_num(var_et_test, nan=1.0, posinf=1.0, neginf=1.0)

        # equally weighted objective function (yield and irrigation)
        if var_irrig_test == 0:  # expect var of 0 for rainfed simulation where irrig depths are 0
            fitness = np.nansum(((y_yield_test - yhat_yield_test) ** 2) * (1 / var_yield_test))
        else:
            fitness = (np.nansum(((y_yield_test - yhat_yield_test) ** 2) * (1 / var_yield_test)) 
                       + np.nansum(((y_irrig_test - yhat_irrig_test) ** 2) * (1 / var_irrig_test))
                       + np.nansum(((y_et_test - yhat_et_test) ** 2) * (1 / var_et_test)))


                
    return fitness


def RunAllCounties(x, *args):
    """ 
    Function to run the AquaCrop model using calibration parameters for multiple counties. 
    
    Arguments:
        x (list): list of calibration values used by the model to simulate crop-water productivity.
        args: Additional parameters passed to the function. Should include the following:
            - default_params (DataFrame): default parameters for the model that are specific to the study area
            - gdd (DataFrame): growing Degree Days data
            - calib (DataFrame): list with calibrated values from first calibration if 2-step calibration is used. Other generate an empty dataframe with variable, value and calib_val columns
            - plnt_date (DataFrame): annual planting dates
            - gridmet_data (DataFrame): meteorological data
            - soil_data (DataFrame): soil data
            - iwc (str): initial water content ('FC': Field Capacity, 'WP': Wilting Point, 'SAT': Saturation
            - sa (list): sensitive paramaters.
            - observed_yield (Dataframe): observed crop yield
            - observed_irrig (DataFrame): observed irrigation data
            - observed_et (DataFrame): Observed evapotranspiration data
            - irrig_method (int): irrigation method
            - start_grwth_period (int): month corresponding to start of the growth period
            - end_grwth_period (int): month corresponding to end of the growth period
            - df_type (str): type of data frame (train or test)

    Returns:

        full (list): list with full (1) crop yield and irrigation and (2) evapotranspiration dataframes for entire simulation period
        train (list): list with train datasets for (1) crop yield and irrigation and (2) evapotranspiration dataframes simulation period
        test (list): list with test datasets for (1) crop yield and irrigation and (2) evapotranspiration dataframes simulation period
        county_flux_df (DataFrame): county evapotranspiration
        county_storage_df (DataFrame): county soil moisture trends
        county_grwth_df (DataFrame): county crop growth
    """
    
    #start_time = time.time()
 # parameters to be calibrated
    param_names = pd.DataFrame(x)
    param_names['rownum'] = param_names.reset_index().index 
    param_names = param_names.rename(columns = {param_names.columns[0]: "value" })
    
    
    default_params, gdd, calib, plnt_date, gridmet_data, soil_data, iwc, sa, observed_yield, observed_irrig, observed_et, irrig_method, start_grwth_period, end_grwth_period, df_type = args # df type is either full, test or train

    
    observed_et = observed_et.copy()
    observed_et['Date'] = pd.to_datetime(observed_et['Date'], format='%Y-%m-%d')
    observed_et['Month'] = observed_et['Date'].dt.month
    observed_et['Year'] = observed_et['Date'].dt.year
    
    
    sa_params = pd.DataFrame(np.array(sa).flatten())
    
    #sa_params = sa_params
    sa_params['rownum'] = sa_params.reset_index().index 
        
    new_names = {
        'tb': 'Tbase', 
        'tu': 'Tupp',
        'ccs': 'SeedSize',
        'den': 'PlantPop',
        'eme': 'Emergence',
        'cgc':'CGC',
        'ccx': 'CCx',
        'senesence': 'Senescence',
        'cdc': 'CDC',
        'mat': 'Maturity',
        'rtm': 'Zmin',
        'flowering': 'Flowering',
        'rtx': 'Zmax',
        'rtshp': 'fshape_r',
        'maxrooting': 'MaxRooting',
        'rtexup': 'SxTopQ',
        'rtexlw': 'SxBotQ',
         #Crop Transpiration
        'kc': 'Kcb',
        'kcdcl': 'fage',
         #Biomass and Yield
        'wp': 'WP',
        'wpy': 'WPy',
        'hi': 'HI0',
        'hipsflo': 'dHI_pre',
        'exc': 'exc',
        'hipsveg': 'a_HI',
        'hingsto': 'b_HI',
        'hinc': 'dHI0',
        'yldform': 'YldForm',
         #Water and Temperature Stress
        'polmn': 'Tmin_up',
        'polmx': 'Tmax_up',
        'pexup': 'p_up',
        'pexlw': 'p_lo',
        'pexshp': 'fshape_w',
        'smt1': 'SMT1',
        'smt2': 'SMT2',
        'smt3': 'SMT3',
        'pstosen': 'pstosen',
        'maxirr': 'MaxIrr',
        'totalirr': 'MaxIrrSeason',
        'anaer': 'Aer',
        'prtshp': 'fshape_ex',
        'psto': 'p_up2',
        'psen': 'p_up3',
        'ppol': 'p_up4',
        'pstoshp': 'fshape_w2',
        'psenshp': 'fshape_w3',
        'ksat': 'Ksat',
        'thetas': 'thS',
        'thfc': 'thFC',
        'thwp': 'thWP',

        
        
        }
    
    # default parameter values for specified crop in given region. If these are not included in the calibration, they are set to the defaults
    SMT1 = default_params['smt1'].item()
    SMT2 = default_params['smt2'].item()
    SMT3 = default_params['smt3'].item()
    MaxIrr = default_params['maxirr'].item()
    MaxIrrSeason = default_params['maxirr_season'].item()
    p_up2 = default_params['p_up2'].item()
    p_up3 = default_params['p_up3'].item()
    CCx = default_params['ccx'].item()
    CDC = default_params['cdc'].item()
    Zmax = default_params['rtx'].item()
    HI0 = default_params['hi'].item()
    WP = default_params['smt1'].item()
    Tmax_up = default_params['polmx'].item()
    Tmax_lo = default_params['polfin'].item()
    
    sa_params[0] = sa_params[0].replace(new_names) # rename the parameters to match the aquacrop params
    sa_params = sa_params.rename(columns = {sa_params.columns[0]: "variable" })   

    sa_params['variable'] = sa_params['variable'].astype(str) # convert to string from TypeError: hasattr(): attribute name must be string
    
    
    sa_params_full = sa_params.merge(param_names) # with both the AquaCrop format and the sa names
    sa_params_full = sa_params_full.drop('rownum', axis = 1)
    
    sa_params_full['calib_val'] = ''
    sa_params_full = pd.DataFrame(sa_params_full)
    
    # rename params from calibration 1
    calib['variable'] = calib['variable'].replace(new_names)

    
    #merge with calib 1 values
    sa_params_full = pd.concat([sa_params_full, calib])
    
    sa_params_full['calib_val'] = pd.to_numeric(sa_params_full['calib_val'], errors='coerce') # add NAN to the calib_val column

    sa_params_full = sa_params_full.reset_index()
    

    for index, row in sa_params_full.iterrows():
        # Check if the variable is 'SMT1' and update the value if it exists in the DataFrame
                if row['variable'] == 'SMT1':
                    if pd.notna(row['calib_val']):
                        SMT1 = row['calib_val']
                        sa_params_full = sa_params_full.drop(index) # drop value from calib
                    #else:
                    elif pd.isna(row['calib_val']):
                        SMT1 = row['value']
                        
                if row['variable'] == 'SMT2':
                    if pd.notna(row['calib_val']):
                        SMT2 = row['calib_val']
                        sa_params_full = sa_params_full.drop(index) # drop value from calib
                    #else:
                    elif pd.isna(row['calib_val']):
                        SMT2 = row['value']

                if row['variable'] == 'SMT3':
                    if pd.notna(row['calib_val']):
                        SMT3 = row['calib_val']
                        sa_params_full = sa_params_full.drop(index) # drop value from calib
                    #else:
                    elif pd.isna(row['calib_val']):
                        SMT3 = row['value']    
                    
                if row['variable'] == 'MaxIrr':
                    if pd.notna(row['calib_val']):
                        MaxIrr = row['calib_val']
                        sa_params_full = sa_params_full.drop(index) # drop value from calib
                    #else:
                    elif pd.isna(row['calib_val']):
                        MaxIrr = row['value']   
                        
                        
                if row['variable'] == 'MaxIrrSeason':
                    if pd.notna(row['calib_val']):
                        MaxIrrSeason = row['calib_val']
                        sa_params_full = sa_params_full.drop(index) # drop value from calib
                    #else:
                    elif pd.isna(row['calib_val']):
                        MaxIrrSeason = row['value']

                if row['variable'] == 'p_up2':
                    if pd.notna(row['calib_val']):
                        p_up2 = row['calib_val']
                        sa_params_full = sa_params_full.drop(index) # drop value from calib
                    elif pd.isna(row['calib_val']):
                        p_up2 = row['value']

                if row['variable'] == 'p_up3':
                    if pd.notna(row['calib_val']):
                        p_up3 = row['calib_val']
                        sa_params_full = sa_params_full.drop(index) # drop value from calib
                    elif pd.isna(row['calib_val']):
                        p_up3 = row['value']

                if row['variable'] == 'CCx':
                    if pd.notna(row['calib_val']):
                        CCx_c = row['calib_val']
                        sa_params_full = sa_params_full.drop(index) # drop value from calib
                    elif pd.isna(row['calib_val']):
                        CCx_c = row['value']

                if row['variable'] == 'CDC':
                    if pd.notna(row['calib_val']):
                        CDC = row['calib_val']
                        sa_params_full = sa_params_full.drop(index) # drop value from calib
                    elif pd.isna(row['calib_val']):
                        CDC = row['value']
                    
                if row['variable'] == 'Zmax':
                    if pd.notna(row['calib_val']):
                        Zmax = row['calib_val']
                        sa_params_full = sa_params_full.drop(index) # drop value from calib
                    elif pd.isna(row['calib_val']):
                        Zmax = row['value']
    
                if row['variable'] == 'HI0':
                    if pd.notna(row['calib_val']):
                        HI0_c = row['calib_val']
                        sa_params_full = sa_params_full.drop(index) # drop value from calib
                    elif pd.isna(row['calib_val']):
                        HI0_c = row['value']
                    
                if row['variable'] == 'WP':
                    if pd.notna(row['calib_val']):
                        WP = row['calib_val']
                        sa_params_full = sa_params_full.drop(index) # drop value from calib
                    elif pd.isna(row['calib_val']):
                        WP = row['value']
                    
                if row['variable'] == 'Tmax_up':
                    if pd.notna(row['calib_val']):
                        Tmax_up = row['calib_val']
                        sa_params_full = sa_params_full.drop(index) # drop value from calib
                    elif pd.isna(row['calib_val']):
                        Tmax_up = row['value']
                    
                if row['variable'] == 'Tmax_lo':
                    if pd.notna(row['calib_val']):
                        Tmax_lo = row['calib_val']
                        sa_params_full = sa_params_full.drop(index) # drop value from calib
                    elif pd.isna(row['calib_val']):
                        Tmax_lo = row['value']
    
    # remaining parameters set to the calib values
    calib_sub = sa_params_full[sa_params_full['calib_val'].notna()] # calib values that do not have defaults
    
    sa_params_full = sa_params_full[sa_params_full['calib_val'].isna()] # parameters to be calibrated
    
    
    # identify the begining of the simulation (year) based on the gridmet data. Used to set the initial soil moisture conditions
    simulation_start = gridmet_data['Year'].min()
    
    while True:
        if p_up2 < p_up3:
            p_up2 = p_up2
            p_up3 = p_up3
        elif p_up2 > p_up3:
            p_up2 = p_up3
            p_up3 = p_up2
        
        irr_mngt = IrrigationManagement(irrigation_method=defaults['irrig_method'].item(),#irrig_method,
                                        SMT = [SMT1,
                                               SMT1,
                                               SMT2,
                                               SMT3],
                                        MaxIrr = MaxIrr,
                                        AppEff = 85,
                                        MaxIrrSeason = MaxIrrSeason)

        # run model for each county + year combination
        calib_counties = []
        counties_grwth = []
        counties_flux = []
        counties_monthly_flux = []
        counties_daily_flux = []
        counties_storage = []
        all_water_flux = []
        daily_flux = []
        model_counties = []
        model_results_canopy = []
        soil_lay_con = [] #list for soil layer theta values
        county_gw = []

        counter_county = -1
        for ids in gridmet_data['crop_mn_yr'].unique(): #crop_mn_yr
                counter_county += 1
                model_results_yield = []
                model_results_grwth = []
                model_results_flux = []
                model_results_flux2 = []
                daily_flux = []
                monthly_flux = []
                model_results_storage = []
                gw_flux_lst = []

                gridmet_df = gridmet_data[gridmet_data['crop_mn_yr'] == ids]
                gridmet_df = gridmet_df.drop_duplicates(subset='Date')

                soil_df = soil_data[soil_data['crop_mn_yr'] == ids]
                
                et_df = observed_et[observed_et['crop_mn_yr'] == ids]
                et_df = et_df['Date'].unique()

                countyname = gridmet_df['crop_mn_codeyear'].str.replace(r'^.*?(?=[a-z, A-Z])', '', regex=True).unique()[0].capitalize()

                counter_year = -1    

                yr = gridmet_df['Year'].unique()[0]
                
                wdf = gridmet_df[['MinTemp', 
                                  'MaxTemp',
                                  'Precipitation', 
                                  'ReferenceET',
                                  'Date'
                         ]]

                
                plnt_date2 = plnt_date[plnt_date['Year']== yr]   

                crop_name = plnt_date2['CropGDD'].item()#[0]
                pdate = plnt_date2['pdate'].tolist()[0][5:]
                hdate = plnt_date2['late_har'].tolist()[0][5:]
                
                #print(crop_name)

                crop = Crop(c_name = crop_name, 
                            Name = crop_name,
                            planting_date = pdate,# remove the additional year that's being added
                            harvest_date = hdate,
                            Emergence = gdd['eme'].item(),
                            Maturity = gdd['mat'].item(),
                            HIstart = gdd['histart'].item(), # remove the additional year that's being added
                            Flowering = gdd['flowering'].item(),
                            MaxRooting = gdd['maxrooting'].item(),
                            Senescence = gdd['senescence'].item(),
                            YldForm = gdd['yldform'].item(),
                            p_up2 = p_up2,
                            p_up3 = p_up3,
                            CGC = CCx/plnt_date2['canopy'].item(),
                            Zmax = Zmax,
                            HI0 = HI0,
                            WP = WP,
                            # temp stress params
                            Tmax_up = Tmax_up,
                            Tmax_lo = Tmax_lo) 
                
        
                

                sa_params_full = sa_params_full[~sa_params_full['variable'].isin(['p_up2',
                                                                                  'p_up3',
                                                                                  'CCx_c',
                                                                                  'CDC',
                                                                                  'Zmax',
                                                                                  'HI0',
                                                                                  'WP'])]

                for index, row in sa_params_full.iterrows():
                    if pd.isna(row['calib_val']):
                           if hasattr(crop, row['variable']):
                                   setattr(crop,
                                           row['variable'],
                                           row['value']) 
                    elif pd.notna(row['calib_val']):
                           if hasattr(crop,
                                      row['variable']):
                                   setattr(crop,
                                           row['variable'],
                                           row['calib_val']) 
                        
                sim_start = f'{yr}/01/01' #dates to match crop data
                sim_end = f'{yr}/12/31'


                weighted_soil = soil_df# returns soil df instead
                weighted_soil = weighted_soil[weighted_soil['Year']== yr]   

                # calculate pedotransfer functions here
                pred_thWP = ((-0.024*((weighted_soil['sand'][0])/100))) + ((0.487*((weighted_soil['clay'][0])/100))) + ((0.006*((weighted_soil['om'][0])/100))) + ((0.005*((weighted_soil['sand'][0])/100))*((weighted_soil['om'][0])/100))- ((0.013*((weighted_soil['clay'][0])/100))*((weighted_soil['om'][0])/100))+ ((0.068*((weighted_soil['sand'][0])/100))*((weighted_soil['clay'][0])/100))+ 0.031
                wp = pred_thWP + (0.14 * pred_thWP) - 0.02
                pred_thFC = ((-0.251*((weighted_soil['sand'][0])/100))) + ((0.195*((weighted_soil['clay'][0])/100)))+ ((0.011*((weighted_soil['om'][0])/100))) + ((0.006*((weighted_soil['sand'][0])/100))*((weighted_soil['om'][0])/100))- ((0.027*((weighted_soil['clay'][0])/100))*((weighted_soil['om'][0])/100))+ ((0.452*((weighted_soil['sand'][0])/100))*((weighted_soil['clay'][0])/100))+ 0.299
                fc = pred_thFC + (1.283 * (np.power(pred_thFC, 2))) - (0.374 * pred_thFC) - 0.015
                                    #fc = pred_thFC + (1.283 * (pred_thFC*pred_thFC)) - (0.374 * pred_thFC) - 0.015
                ts =weighted_soil["theta_s"][0]
                ks =(weighted_soil['ksat'][0])*240

                # update the soil parameters in the input params df 
                for index, row in sa_params_full.iterrows():
                    if row['variable'] == 'Ksat':
                          Ksat = row['value']
                    elif 'Ksat' not in sa_params_full.values:
                          Ksat = ks # soi

                    if row['variable'] == 'thS':
                          thS = row['value']
                    elif 'thS' not in sa_params_full.values:
                          thS = ts

                    if row['variable'] == 'thWP':
                          thWP = row['value']
                    elif 'thWP' not in sa_params_full.values:
                          thWP = wp

                    if row['variable'] == 'thFC':
                          thFC = row['value']
                    elif 'thFC' not in sa_params_full.values:
                          thFC = fc

                # create custom soil layer
                custom_soil = Soil('custom',
                                   cn = 61,
                                   rew=7,
                                   dz=[0.1]*12)

                custom_soil.add_layer(thickness=2,
                                      thS=ts, # assuming soil properties are the same in the upper 0.1m
                                      Ksat=ks,
                                      thWP =wp,
                                      thFC = fc,
                                      penetrability = 100.0)

               
                # run model
                if yr == simulation_start: # at the begining of the simulation period, specify the initial soil moisture
                    initWC = InitialWaterContent(value=[str(iwc)])
                elif yr > simulation_start:
                    initWC = soil_lay_con[0]
                

                model_c = AquaCropModel(sim_start_time= sim_start,
                                        sim_end_time = sim_end,
                                        weather_df = wdf,
                                        soil = custom_soil,
                                        crop = crop,
                                        initial_water_content = initWC, 
                                        irrigation_management= irr_mngt,
                                        off_season = True)
  
                model_c.run_model(till_termination=True)
                #model_c.run_model(initialize_model=True)
                #model_c_et = model_c._outputs.water_flux
                model_c_irr = model_c._outputs.final_stats
                model_c_water_storage = model_c._outputs.water_storage
                model_c_crp_grwth = model_c._outputs.crop_growth
                
                #print(yr)

                model_c_irr = model_c_irr.rename(columns={
                                                       'Yield (tonne/ha)': 'Calib Yield (t/ha)',
                                                       'Seasonal irrigation (mm)': 'Calib Irrigation (mm)'
                                                       })           
                            
                model_c_irr['USDA Harvest Date'] = plnt_date2['har'].tolist()[0] # include the USDA harvest to compare with the model harvest date
                model_c_irr['Harvest Date (YYYY/MM/DD)'] = pd.to_datetime(model_c_irr['Harvest Date (YYYY/MM/DD)'],
                                                                          format='%Y-%m-%d')

                # Convert the 'date' column back to a new column in YMD format
                model_c_irr = model_c_irr.assign(Year =  model_c_irr['Harvest Date (YYYY/MM/DD)'].dt.year)
                model_results_yield.append(model_c_irr)
                
                model_results_df = pd.concat(model_results_yield)
                model_results_df['County'] = countyname

                
                #print(model_results_df)
                # compare model and observed yield and irrigation (merge dataframes)
                

                # Drop rows where 'YieldUSDA' column is NaN
                #model_commodel_comp_one.dropna(subset=['all_yield'], inplace=True)

                model_comp = model_results_df.copy()
                
                 # convert to t/ha

                #print(model_comp)
                model_comp = model_comp.dropna()
                model_counties.append(model_comp)

                model_counties_df = pd.concat(model_counties)
                
                observed = observed_yield.merge(observed_irrig, on=['Year', 'County', 'crop Type'],
                                     how='outer')
                
                observed = observed.assign(YieldUSDA = observed['all_yield']*0.0673)
                #print(observed)
                
                model_counties_df = model_counties_df.merge(observed,
                          on=['Year', 'County', 'crop Type'],
                                     how='inner')
                
                 
                # shuffle data and generate random
                #random.shuffle(model_counties_df)
                model_counties_df = model_counties_df.sample(frac=1, random_state=42)
                yi_train = model_counties_df[:int((len(model_counties_df)+1)*.80)] #Remaining 80% to training set
                yi_test = model_counties_df[int((len(model_counties_df)+1)*.80):] # test
                
                # identify train and test years
                train_yrs = yi_train[['County', 'Year']].drop_duplicates()
                test_yrs = yi_test[['County', 'Year']].drop_duplicates()
                
                
                # soil water storage
                model_c_water_storage.iloc[-1] = model_c_water_storage.iloc[-2].values
                model_c_water_storage['Date'] = wdf['Date'].values
                model_c_water_storage['County'] = countyname
                

                soil_water = model_c_water_storage.copy()
                #soil_water = soil_water.tail() dec 31 has 0, so use Dec 30 
                soil_water = soil_water.iloc[-1]

                soil_water2 = (soil_water[['th1',
                                           'th2',
                                           'th3',
                                           'th4',
                                           'th5',
                                           'th6',
                                           'th7',
                                           'th8',
                                           'th9',
                                           'th10',
                                           'th11',
                                           'th12']]).values.tolist()

                # calculate the initial soil moisture conditions in the soil layers for the first day of subsequent year
                initWC = InitialWaterContent(wc_type = 'Num',  
                                             method = 'Depth', 
                                             depth_layer= [0.05,
                                                           0.15,
                                                           0.25,
                                                           0.35,
                                                           0.45,
                                                           0.55,
                                                           0.65,
                                                           0.75,
                                                           0.85,
                                                           0.95,
                                                           1.05,
                                                           1.15],
                                             value = soil_water2)
                soil_lay_con.clear() # remove last results from list
                soil_lay_con.append(initWC) # save new theta values
                
                # create df with the daily soil moisture values during simulation period
                model_results_storage.append(model_c_water_storage)
                model_storage_df = pd.concat(model_results_storage)
                                    
                
                counties_storage.append(model_storage_df)
                county_storage_df = pd.concat(counties_storage)
                
                
                # ET
                model_c_et['Date'] = wdf['Date'].values
                model_c_et['Year'] = yr
                model_c_et['Month'] = model_c_et['Date'].dt.month  
                
                # make a copy of daily water
                gw_flux = model_c_et.copy()
                gw_flux_lst.append(gw_flux)
                gw_flux_df = pd.concat(gw_flux_lst)
                gw_flux_df['County'] = countyname
                
                county_gw.append(gw_flux_df)
                county_gw_df = pd.concat(county_gw)
                
                model_c_et = model_c_et[(model_c_et['Month'] < end_grwth_period) & (model_c_et['Month'] > start_grwth_period)] # filter for corn growth period
                        #start_grwth_period, end_grwth_period            
                  
                model_results_flux.append(model_c_et)
                                    
                model_flux_df = pd.concat(model_results_flux)
                model_flux_df['County'] = countyname
                
                model_flux_df['ET'] = model_flux_df['Es'] + model_flux_df['Tr']
                model_flux_df['Date'] = model_flux_df['Year'].astype(str)  + '-'  + model_flux_df['Month'].astype(str) + '-' + '01'
                model_flux_df['Date'] = pd.to_datetime(model_flux_df['Date'],
                                                       format='%Y-%m-%d')
                model_flux_df['Month'] = model_flux_df['Date'].dt.month 
                                    
                model_flux_df = model_flux_df[['Date',
                                               'Month',
                                               'Year',
                                               'County',
                                               'ET']]
                model_flux_df = model_flux_df.groupby(['County',
                                                       'Month',
                                                       'Year'],
                                                      as_index=False)[['ET']].sum()
                
                counties_flux.append(model_flux_df)
                county_flux_df = pd.concat(counties_flux)
                                   
                #county_flux_df = county_flux_df.merge(observed_et, 
                                                      #how='outer', 
                                                      #on=['County',
                                                          #'Month',
                                                          #'Year']
                                                     #)
                                    
                # return copy of et df to see monthly data
                annual_model_flux = county_flux_df.copy()
                
                annual_flux = annual_model_flux.groupby(['County', 'Year'],
                                                        as_index=False)[['ET']].sum()
                
                
                county_flux_train = annual_flux[['County', 'Year']].apply(tuple, axis=1)
                train_yrs = train_yrs.apply(tuple, axis=1)


                et_train = annual_flux[county_flux_train.isin(train_yrs)]

                # test
                county_flux_test = annual_flux[['County', 'Year']].apply(tuple, axis=1)
                test_yrs = test_yrs.apply(tuple,
                                          axis=1)
                et_test = annual_flux[county_flux_test.isin(test_yrs)]
                
                full = [model_counties_df, annual_flux]
                train = [yi_train, et_train]
                test = [yi_test, et_test]
                
                
                # crop growth
                model_c_crp_grwth['Date'] = wdf['Date'].values
                model_results_grwth.append(model_c_crp_grwth)
                model_grwth_df = pd.concat(model_results_grwth)
                model_grwth_df['County'] = countyname
                
                counties_grwth.append(model_grwth_df)
                county_grwth_df = pd.concat(counties_grwth)
                county_grwth_df['Year'] = county_grwth_df['Date'].dt.year
                
                
                              
        return full, train, test, county_flux_df, county_storage_df, county_gw_df, county_grwth_df
       

def RunModelBiasCorrected(default_params, plnt_date, gridmet_data, soil_data, bias_correction_params, start_grwth_period, end_grwth_period):
    """ 
    Function to run the AquaCrop model using calibration parameters for multiple counties. 
    
    Arguments:
        x (list): list of calibration values used by the model to simulate crop-water productivity.
        args: Additional parameters passed to the function. Should include the following:
            - default_params (DataFrame): default parameters for the model that are specific to the study area
            - plnt_date (DataFrame): annual planting dates
            - gridmet_data (DataFrame): meteorological data
            - soil_data (DataFrame): soil data
            - start_grwth_period (int): month corresponding to start of the growth period
            - end_grwth_period (int): month corresponding to end of the growth period


    Returns:

        model_counties_df (DataFrame): simulated yield and irrigation including the bias corrected values
        county_flux_df (DataFrame): county evapotranspiration
        county_storage_df (DataFrame): county soil moisture trends
        county_gw_df (DataFrame): water balance quantities
        county_grwth_df (DataFrame): county crop growth
    """
    
    #start_time = time.time()
 # parameters to be calibrated
    
    
    
    #default_params, plnt_date, gridmet_data, soil_data,  start_grwth_period, end_grwth_period = args # df type is either full, test or train

    
    p_up2 = default_params['p_up2'].item()
    p_up3 = default_params['p_up3'].item()
    
    # identify the begining of the simulation (year) based on the gridmet data. Used to set the initial soil moisture conditions
    simulation_start = gridmet_data['Year'].min()
    
    while True:
        if p_up2 < p_up3:
            p_up2 = p_up2
            p_up3 = p_up3
        elif p_up2 > p_up3:
            p_up2 = p_up3
            p_up3 = p_up2
        
        irr_mngt = IrrigationManagement(irrigation_method=default_params['irrig_method'].item(),#irrig_method,
                                        SMT = [default_params['smt1'].item(),
                                               default_params['smt1'].item(),
                                               default_params['smt2'].item(),
                                               default_params['smt3'].item()],
                                        MaxIrr = default_params['maxirr'].item(),
                                        AppEff = 85,
                                        MaxIrrSeason = default_params['maxirr_season'].item())

        # run model for each county + year combination
        calib_counties = []
        counties_grwth = []
        counties_flux = []
        counties_monthly_flux = []
        counties_daily_flux = []
        counties_storage = []
        all_water_flux = []
        daily_flux = []
        model_counties = []
        model_results_canopy = []
        soil_lay_con = [] #list for soil layer theta values
        county_gw = []

        counter_county = -1
        for ids in gridmet_data['crop_mn_yr'].unique(): #crop_mn_yr is the unique id
                counter_county += 1
                model_results_yield = []
                model_results_grwth = []
                model_results_flux = []
                model_results_flux2 = []
                daily_flux = []
                monthly_flux = []
                model_results_storage = []
                gw_flux_lst = []

                gridmet_df = gridmet_data[gridmet_data['crop_mn_yr'] == ids]
                gridmet_df = gridmet_df.drop_duplicates(subset='Date')

                soil_df = soil_data[soil_data['crop_mn_yr'] == ids]
                
                #et_df = observed_et[observed_et['crop_mn_yr'] == ids]
                #et_df = et_df['Date'].unique()

                countyname = gridmet_df['crop_mn_codeyear'].str.replace(r'^.*?(?=[a-z, A-Z])', '', regex=True).unique()[0].capitalize()

                counter_year = -1    

                yr = gridmet_df['Year'].unique()[0]
                
                wdf = gridmet_df[['MinTemp', 
                                  'MaxTemp',
                                  'Precipitation', 
                                  'ReferenceET',
                                  'Date'
                         ]]

                
                plnt_date2 = plnt_date[plnt_date['Year']== yr]   

                crop_name = plnt_date2['CropGDD'].item()#[0]
                pdate = plnt_date2['pdate'].tolist()[0][5:]
                hdate = plnt_date2['late_har'].tolist()[0][5:]
                

                crop = Crop(c_name = crop_name, 
                            Name = crop_name,
                            planting_date = pdate,# remove the additional year that's being added
                            harvest_date = hdate,
                            Emergence = default_params['eme'].item(),
                            Maturity = default_params['mat'].item(),
                            HIstart = default_params['histart'].item(), # remove the additional year that's being added
                            Flowering = default_params['flowering'].item(),
                            MaxRooting = default_params['maxrooting'].item(),
                            Senescence = default_params['senescence'].item(),
                            YldForm = default_params['yldform'].item(),
                            CCx = default_params['ccx'].item(),
                            CGC = default_params['ccx'].item()/plnt_date2['canopy'].item(),
                            Zmax = default_params['rtx'].item(),
                            HI0 = default_params['hi'].item(),
                            WP = default_params['wp'].item(),
                            Kcb = default_params['kc'].item(),
                            a_HI = default_params['hipsveg'].item(),
                            Tmax_up = default_params['polmx'].item(),
                            Tmax_lo = default_params['polfin'].item(),
                            p_up2 = default_params['p_up2'].item(),
                            p_up3 = default_params['p_up3'].item())
                
                        
                sim_start = f'{yr}/01/01' #dates to match crop data
                sim_end = f'{yr}/12/31'


                weighted_soil = soil_df# returns soil df instead
                weighted_soil = weighted_soil[weighted_soil['Year']== yr]   

                # calculate pedotransfer functions here
                pred_thWP = ((-0.024*((weighted_soil['sand'][0])/100))) + ((0.487*((weighted_soil['clay'][0])/100))) + ((0.006*((weighted_soil['om'][0])/100))) + ((0.005*((weighted_soil['sand'][0])/100))*((weighted_soil['om'][0])/100))- ((0.013*((weighted_soil['clay'][0])/100))*((weighted_soil['om'][0])/100))+ ((0.068*((weighted_soil['sand'][0])/100))*((weighted_soil['clay'][0])/100))+ 0.031
                wp = pred_thWP + (0.14 * pred_thWP) - 0.02
                pred_thFC = ((-0.251*((weighted_soil['sand'][0])/100))) + ((0.195*((weighted_soil['clay'][0])/100)))+ ((0.011*((weighted_soil['om'][0])/100))) + ((0.006*((weighted_soil['sand'][0])/100))*((weighted_soil['om'][0])/100))- ((0.027*((weighted_soil['clay'][0])/100))*((weighted_soil['om'][0])/100))+ ((0.452*((weighted_soil['sand'][0])/100))*((weighted_soil['clay'][0])/100))+ 0.299
                fc = pred_thFC + (1.283 * (np.power(pred_thFC, 2))) - (0.374 * pred_thFC) - 0.015
                                    #fc = pred_thFC + (1.283 * (pred_thFC*pred_thFC)) - (0.374 * pred_thFC) - 0.015
                ts =weighted_soil["theta_s"][0]
                ks =(weighted_soil['ksat'][0])*240

                
                # create custom soil layer
                custom_soil = Soil('custom',
                                   cn = 61,
                                   rew=7,
                                   dz=[0.1]*12)

                custom_soil.add_layer(thickness=2,
                                      thS=ts, # assuming soil properties are the same in the upper 0.1m
                                      Ksat=ks,
                                      thWP =wp,
                                      thFC = fc,
                                      penetrability = 100.0)

               
                # run model
                if yr == simulation_start: # at the begining of the simulation period, specify the initial soil moisture
                    initWC = InitialWaterContent(value=[str(default_params['init_water'].item())])
                elif yr > simulation_start:
                    initWC = soil_lay_con[0]
                            

                model_c = AquaCropModel(sim_start_time= sim_start,
                                        sim_end_time = sim_end,
                                        weather_df = wdf,
                                        soil = custom_soil,
                                        crop = crop,
                                        initial_water_content = initWC, 
                                        irrigation_management= irr_mngt,
                                        off_season = True)
  
                model_c.run_model(till_termination=True)
                #model_c.run_model(initialize_model=True)
                model_c_et = model_c._outputs.water_flux
                model_c_irr = model_c._outputs.final_stats
                model_c_water_storage = model_c._outputs.water_storage
                model_c_crp_grwth = model_c._outputs.crop_growth
                
     
                # yield and irrigation
                model_c_irr = model_c_irr.rename(columns={
                                                       'Yield (tonne/ha)': 'Calib Yield (t/ha)',
                                                       'Seasonal irrigation (mm)': 'Calib Irrigation (mm)'
                                                       })           
                            
                model_c_irr['USDA Harvest Date'] = plnt_date2['har'].tolist()[0] # include the USDA harvest to compare with the model harvest date
                model_c_irr['Harvest Date (YYYY/MM/DD)'] = pd.to_datetime(model_c_irr['Harvest Date (YYYY/MM/DD)'],
                                                                          format='%Y-%m-%d')

                # Convert the 'date' column back to a new column in YMD format
                model_c_irr = model_c_irr.assign(Year =  model_c_irr['Harvest Date (YYYY/MM/DD)'].dt.year)
                model_results_yield.append(model_c_irr)
                
                model_results_df = pd.concat(model_results_yield)
                model_results_df['County'] = countyname
                model_comp = model_results_df.copy()
                model_comp = model_comp.dropna()
                model_counties.append(model_comp)
                model_counties_df = pd.concat(model_counties)
                
                # bias correction
                bias_correction_params = bias_correction_params[bias_correction_params['Crop'] == crop_name] # filter bias corrction df for the crop of choice used in model
                model_counties_df['Bias Corrected Yield (t/ha)'] = model_counties_df['Calib Yield (t/ha)'] + (bias_correction_params['yield_m'].item() +(bias_correction_params['yield_c'].item()*model_counties_df['Calib Yield (t/ha)']))
                model_counties_df['Bias Corrected Irrigation (mm'] = model_counties_df['Calib Irrigation (mm)'] + (bias_correction_params['irrig_m'].item() +(bias_correction_params['irrig_c'].item()*model_counties_df['Calib Irrigation (mm)']))
                
                
                # soil water storage
                model_c_water_storage.iloc[-1] = model_c_water_storage.iloc[-2].values
                model_c_water_storage['Date'] = wdf['Date'].values
                model_c_water_storage['County'] = countyname
                

                soil_water = model_c_water_storage.copy()
                #soil_water = soil_water.tail() dec 31 has 0, so use Dec 30 
                soil_water = soil_water.iloc[-1]

                soil_water2 = (soil_water[['th1',
                                           'th2',
                                           'th3',
                                           'th4',
                                           'th5',
                                           'th6',
                                           'th7',
                                           'th8',
                                           'th9',
                                           'th10',
                                           'th11',
                                           'th12']]).values.tolist()

                # calculate the initial soil moisture conditions in the soil layers for the first day of subsequent year
                initWC = InitialWaterContent(wc_type = 'Num',  
                                             method = 'Depth', 
                                             depth_layer= [0.05,
                                                           0.15,
                                                           0.25,
                                                           0.35,
                                                           0.45,
                                                           0.55,
                                                           0.65,
                                                           0.75,
                                                           0.85,
                                                           0.95,
                                                           1.05,
                                                           1.15],
                                             value = soil_water2)
                soil_lay_con.clear() # remove last results from list
                soil_lay_con.append(initWC) # save new theta values
                
                # create df with the daily soil moisture values during simulation period
                model_results_storage.append(model_c_water_storage)
                model_storage_df = pd.concat(model_results_storage)
                                    
                
                counties_storage.append(model_storage_df)
                county_storage_df = pd.concat(counties_storage)
                
                
                # ET
                model_c_et['Date'] = wdf['Date'].values
                model_c_et['Year'] = yr
                model_c_et['Month'] = model_c_et['Date'].dt.month  
                
                # make a copy of daily water
                gw_flux = model_c_et.copy()
                gw_flux_lst.append(gw_flux)
                gw_flux_df = pd.concat(gw_flux_lst)
                gw_flux_df['County'] = countyname
                
                county_gw.append(gw_flux_df)
                county_gw_df = pd.concat(county_gw)
                
                model_c_et = model_c_et[(model_c_et['Month'] < end_grwth_period) & (model_c_et['Month'] > start_grwth_period)] # filter for corn growth period
                        #start_grwth_period, end_grwth_period            
                  
                model_results_flux.append(model_c_et)
                                    
                model_flux_df = pd.concat(model_results_flux)
                model_flux_df['County'] = countyname
                
                model_flux_df['ET'] = model_flux_df['Es'] + model_flux_df['Tr']
                model_flux_df['Date'] = model_flux_df['Year'].astype(str)  + '-'  + model_flux_df['Month'].astype(str) + '-' + '01'
                model_flux_df['Date'] = pd.to_datetime(model_flux_df['Date'],
                                                       format='%Y-%m-%d')
                model_flux_df['Month'] = model_flux_df['Date'].dt.month 
                                    
                model_flux_df = model_flux_df[['Date',
                                               'Month',
                                               'Year',
                                               'County',
                                               'ET']]
                model_flux_df = model_flux_df.groupby(['County',
                                                       'Month',
                                                       'Year'],
                                                      as_index=False)[['ET']].sum()
                
                counties_flux.append(model_flux_df)
                county_flux_df = pd.concat(counties_flux)
                                   

                                    
                # return copy of et df to see monthly data
                annual_model_flux = county_flux_df.copy()
                
                annual_flux = annual_model_flux.groupby(['County', 'Year'],
                                                        as_index=False)[['ET']].sum()
                
                
                
                
                # crop growth
                model_c_crp_grwth['Date'] = wdf['Date'].values
                model_results_grwth.append(model_c_crp_grwth)
                model_grwth_df = pd.concat(model_results_grwth)
                model_grwth_df['County'] = countyname
                
                counties_grwth.append(model_grwth_df)
                county_grwth_df = pd.concat(counties_grwth)
                county_grwth_df['Year'] = county_grwth_df['Date'].dt.year
                
                
                              
        return  model_counties_df #, county_flux_df, county_storage_df, county_gw_df, county_grwth_df (uncomment to return the ET, soil compartment moisture, water balance and crop growth dfs
       
       
 

def PsoResults(pso_list, df_number, *args):
    """
    Function to label model results as either the train or test datasets. Dataframe numbers from RunAllCounties() are used.
    
    Arguments:
        pso_list (list): list with results fromparticle sawrm optimization iterations.
        df_number (int): dataframe number from runAllCounties(). Train: 1, Test:2
        args: Additional parameters passed to the function. Should include the following:
            - default_params (DataFrame): default parameters for the model that are specific to the study area
            - gdd (DataFrame): growing Degree Days data
            - calib (DataFrame): list with calibrated values from first calibration if 2-step calibration is used. Other generate an empty dataframe with variable, value and calib_val columns
            - plnt_date (DataFrame): annual planting dates
            - gridmet_data (DataFrame): meteorological data
            - soil_data (DataFrame): soil data
            - iwc (str): initial water content ('FC': Field Capacity, 'WP': Wilting Point, 'SAT': Saturation
            - sa (list): sensitive paramaters.
            - observed_yield (Dataframe): observed crop yield
            - observed_irrig (DataFrame): observed irrigation data
            - observed_et (DataFrame): Observed evapotranspiration data
            - irrig_method (int): irrigation method
            - start_grwth_period (int): month corresponding to start of the growth period
            - end_grwth_period (int): month corresponding to end of the growth period
            - df_type (str): type of data frame (train or test
    Returns:

        irrig_full_df (DataFrame): AquaCrop yield-irrigation model results
    """
    
    default_params, gdd, calib, plnt_date, gridmet_data, soil_data, iwc, sa, observed_yield, observed_irrig, observed_et, irrig_method, start_grwth_period, end_grwth_period, df_type = args
    
    irrig_list = []
    for index, item in enumerate(pso_list):
    
        irrig_df = RunAllCounties(item,
                                    default_params,
                                    gdd,
                                    calib,
                                    plnt_date,
                                    gridmet_data,
                                    soil_data,
                                    iwc,
                                    sa,
                                    observed_yield,
                                    observed_irrig,
                                    observed_et,
                                    irrig_method,
                                    start_grwth_period,
                                    end_grwth_period,
                                    df_type)[df_number][0] 


        if df_number == 1:
            irrig_df = irrig_df.rename(columns = {'Calib Irrigation (mm)': 'irrigation_train'})
            irrig_df = irrig_df.rename(columns = {'Calib Yield (t/ha)': 'yield_train'})
        elif df_number == 2:
            irrig_df = irrig_df.rename(columns = {'Calib Irrigation (mm)': 'irrigation_test'})
            irrig_df = irrig_df.rename(columns = {'Calib Yield (t/ha)': 'yield_test'})
            
        
        irrig_df['iteration'] = index + 1
        irrig_df['initWC'] = iwc
        irrig_list.append(irrig_df)
        irrig_full_df = pd.concat(irrig_list)
        
    return irrig_full_df  
       
        

def PsoIteration(lb, ub, iteration_count, minfunc, swarmsize, maxiter, *args):
    
    default_params, gdd, calib, plnt_date, gridmet_data, soil_data, iwc, sa, observed_yield, observed_irrig, observed_et, irrig_method, start_grwth_period, end_grwth_period, df_type = args

    xopt_wp, fopt_wp = pso(FitnessCalc,
                           lb,
                           ub,
                           minfunc=minfunc,
                           swarmsize=swarmsize,
                           maxiter=maxiter,
                           args=(default_params,
                                 gdd,
                                 calib,
                                 plnt_date,
                                 gridmet_data,
                                 soil_data,
                                 iwc,
                                 sa,
                                 observed_yield,
                                 observed_irrig,
                                 observed_et,
                                 irrig_method,
                                 start_grwth_period,
                                 end_grwth_period,
                                 df_type
                                )
                          )

    calib_df = pd.DataFrame(data={
        'iteration': iteration_count,
        'initWC': iwc,
        'variable': sa.flatten(),  # Assuming influ_var is a global variable or is defined elsewhere
        'value': xopt_wp,
        'obj_fun': fopt_wp,
        'set_minfunc': minfunc,
        'set_swarmsize': swarmsize,
        'set_maxiter': maxiter
    })

    return calib_df
        

def nrmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """ Normalized Root Mean Squared Error """
    # Remove NaNs
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    # Calculate RMSE
    rmse_value = rmse(actual_clean, predicted_clean)
    
    # Calculate range of actual values
    range_actual = np.max(actual_clean) - np.min(actual_clean)
    
    # Check if range_actual is zero to avoid division by zero
    if range_actual == 0:
        raise ValueError("The range of 'actual' values is zero, cannot normalize RMSE.")
    
    # Calculate NRMSE
    nrmse_value = rmse_value / range_actual
    
    return nrmse_value
    
    