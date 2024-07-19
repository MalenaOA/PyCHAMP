import pandas as pd
import numpy as np



#class SoilWrangling:
    
    #def __init__(self,x,y,z,W,D):
    #def __init__(self, soil_list):
        #self.soil_list = soil_list
    
def SoilCounty(soil_list):
        custom_soil = []
        soil = soil_list
        
        for df in soil:

                    #soils_df = df.groupby(['Year'])
            soils_df = [v for k, v in df.groupby('Year')]
            soils = []
            for j in soils_df:
                soils_df_pivot = j.drop(['crop_mn_codeyear', 'Year'], axis=1)
                soils_df_pivot['thickness'] = np.where(
                                         soils_df_pivot['depth'] == '0_5', '5',
                            np.where(soils_df_pivot['depth'] == '5_15', '10',
                                     np.where(soils_df_pivot['depth'] == '15_30', '15',
                                              np.where(soils_df_pivot['depth'] == '30_60', '30',
                                                       np.where(soils_df_pivot['depth'] == '60_100', '40', '100')
                                                               )
                                                      )
                                             )
                                    )

                soils_df_pivot['thickness'] = soils_df_pivot['thickness'].astype(float)

                weighted_soil = soils_df_pivot.copy()

                        #, 'clay', 'hb', 'ksat', 'lambda', 'n', 'om', 'sand', 'silt', 'theta_r', 'theta_s'
                weighted_soil['alpha'] = weighted_soil['alpha'] * weighted_soil['thickness']
                weighted_soil['clay'] = weighted_soil['clay'] * weighted_soil['thickness']
                weighted_soil['hb'] =  weighted_soil['hb'] * weighted_soil['thickness']
                weighted_soil['ksat'] = weighted_soil['ksat'] * weighted_soil['thickness']
                weighted_soil['lambda'] = weighted_soil['lambda'] * weighted_soil['thickness']
                weighted_soil['n'] = weighted_soil['n'] * weighted_soil['thickness']
                weighted_soil['om'] = weighted_soil['om'] * weighted_soil['thickness']
                weighted_soil['sand'] = weighted_soil['sand'] * weighted_soil['thickness']
                weighted_soil['silt'] = weighted_soil['silt'] * weighted_soil['thickness']
                weighted_soil['theta_r'] = weighted_soil['theta_r'] * weighted_soil['thickness']
                weighted_soil['theta_s'] = weighted_soil['theta_s'] * weighted_soil['thickness']


                        # drop dept
                weighted_soil = weighted_soil.drop('depth', axis = 1)
                weighted_soil['rand'] = 'rand'

                weighted_soil = weighted_soil.groupby('rand').sum()/200  

                pred_thWP = ((-0.024*((weighted_soil['sand'][0])/100))) + ((0.487*((weighted_soil['clay'][0])/100))) + ((0.006*((weighted_soil['om'][0])/100))) + ((0.005*((weighted_soil['sand'][0])/100))*((weighted_soil['om'][0])/100))- ((0.013*((weighted_soil['clay'][0])/100))*((weighted_soil['om'][0])/100))+ ((0.068*((weighted_soil['sand'][0])/100))*((weighted_soil['clay'][0])/100))+ 0.031
                wp = pred_thWP + (0.14 * pred_thWP) - 0.02
                pred_thFC = ((-0.251*((weighted_soil['sand'][0])/100))) + ((0.195*((weighted_soil['clay'][0])/100)))+ ((0.011*((weighted_soil['om'][0])/100))) + ((0.006*((weighted_soil['sand'][0])/100))*((weighted_soil['om'][0])/100))- ((0.027*((weighted_soil['clay'][0])/100))*((weighted_soil['om'][0])/100))+ ((0.452*((weighted_soil['sand'][0])/100))*((weighted_soil['clay'][0])/100))+ 0.299
                fc = pred_thFC + (1.283 * (np.power(pred_thFC, 2))) - (0.374 * pred_thFC) - 0.015
                        #fc = pred_thFC + (1.283 * (pred_thFC*pred_thFC)) - (0.374 * pred_thFC) - 0.015
                ts =weighted_soil["theta_s"][0]
                ks =(weighted_soil['ksat'][0])*240

                        # new soils df with the params to add to SA
                variables = pd.DataFrame(('ksat', 'thetas', 'thfc', 'thwp'), columns = ['var']) # changed from +/-20% to +/-5%
                lower_b = pd.DataFrame(((ks-(ks*0.05)), (ts-(ts*0.05)), 
                                                (fc-(fc*0.05)), (wp-(wp*0.05))), columns = ['lb'])
                upper_b = pd.DataFrame(((ks+(ks*0.05)), (ts+(ts*0.05)), 
                                                (fc+(fc*0.05)), (wp+(wp*0.05))), columns = ['ub'])


                soil_params = pd.concat([variables, lower_b, upper_b], axis = 1)  
                
                soils.append(soil_params)

        return(soils)
    
    
    
def SoilCompart(soilList):
    """
    This function is used the calculate the custom soil layers for each county for any given crop-irrigation 
    management combination. Soil properties are depth weighted
    
    Arguments:
        soilList (list): list of dataframes (grouped by location/county) containing annual soil charecteristics for the following depths:
        0-5 mm, 5-15 mm, 30-60 mm, 60-100 mm, 100-200 mm
        
    Returns:
    
    custom_soil (list): list with dataframes of the annual depth weighted soil characteristics for each location
       
    """
    
    custom_soil = []

    for df in soilList:
        
        #soils_df = df.groupby(['Year'])
        #soils_df = [v for k, v in df.groupby('Year')]
            
        soils = []
        #print(soils_df)
        
        #df['Year'] = df['Year'].astype(int)
            #soils_df_pivot = j.drop(['crop_mn_codeyear', 'Year'], axis=1)
        for ids in df['crop_mn_yr'].unique():
            soils_df_pivot = df[df['crop_mn_yr'] == ids]
            soils_df_pivot['thickness'] = np.where(
                             soils_df_pivot['depth'] == '0_5', '5',
                np.where(soils_df_pivot['depth'] == '5_15', '10',
                         np.where(soils_df_pivot['depth'] == '15_30', '15',
                                  np.where(soils_df_pivot['depth'] == '30_60', '30',
                                           np.where(soils_df_pivot['depth'] == '60_100', '40', '100')
                                                   )
                                          )
                                 )
                        )
            
            soils_df_pivot['thickness'] = soils_df_pivot['thickness'].astype(float)
            
            weighted_soil = soils_df_pivot.copy()
            
            #, 'clay', 'hb', 'ksat', 'lambda', 'n', 'om', 'sand', 'silt', 'theta_r', 'theta_s'
            weighted_soil['alpha'] = weighted_soil['alpha'] * weighted_soil['thickness']
            weighted_soil['clay'] = weighted_soil['clay'] * weighted_soil['thickness']
            weighted_soil['hb'] =  weighted_soil['hb'] * weighted_soil['thickness']
            weighted_soil['ksat'] = weighted_soil['ksat'] * weighted_soil['thickness']
            weighted_soil['lambda'] = weighted_soil['lambda'] * weighted_soil['thickness']
            weighted_soil['n'] = weighted_soil['n'] * weighted_soil['thickness']
            weighted_soil['om'] = weighted_soil['om'] * weighted_soil['thickness']
            weighted_soil['sand'] = weighted_soil['sand'] * weighted_soil['thickness']
            weighted_soil['silt'] = weighted_soil['silt'] * weighted_soil['thickness']
            weighted_soil['theta_r'] = weighted_soil['theta_r'] * weighted_soil['thickness']
            weighted_soil['theta_s'] = weighted_soil['theta_s'] * weighted_soil['thickness']
            
            
            # drop dept
            weighted_soil = weighted_soil.drop('depth', axis = 1)
           
        #weighted_soil['rand'] = 'rand'
            weighted_soil.reset_index(inplace=True)
            weighted_soil = weighted_soil.groupby(['crop_mn_yr', 'Year', 'crop_mn_codeyear']).sum()/200  
            weighted_soil.reset_index(inplace=True)
     
            weighted_soil.drop_duplicates('crop_mn_yr')  
            
            soils.append(weighted_soil)
            soils_final =  pd.concat(soils)               
        custom_soil.append(soils_final)

                    #print(custom)

    return(custom_soil)