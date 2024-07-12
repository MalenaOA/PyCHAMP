import os
import numpy as np
#import py_champ
from py_champ.models.particle_swarm import GlobalBestPSO
from py_champ.utility.util import TimeRecorder

# =============================================================================
# General settings
# =============================================================================
exp_dir = r"C:/Users/cecil/Dropbox/phD_Cecilia/Projects/Crop Insurance ERL/Codes/experiments/cali2"
#exp_dir = r"C:\Users\CL\Desktop\SD6Model_1f1w\sd6_1f1w_cali_1"
os.chdir(exp_dir)

def obj_func(x, seeds, exp_dir, **kwargs):
    # Need to be placed here to use dill in joblib
    import os
    import sys
    import dill
    import numpy as np
    import pandas as pd
    from joblib.externals.loky import set_loky_pickler
    from py_champ.models.sd6_model_1f1w_ci import SD6Model_1f1w_ci
    join = os.path.join
    wd = r"C:/Users/cecil/Dropbox/phD_Cecilia/Projects/Crop Insurance ERL/Codes/experiments/Data"
    sys.setrecursionlimit(10000)
    set_loky_pickler("dill")    # joblib

    # Load inputs and data
    with open(join(wd, "Inputs", "Inputs_SD6_2012_2022_ci_test2011.pkl"), "rb") as f:
        (aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict,
         prec_aw_step, crop_price_step, harvest_price_step, projected_price_step,
         aph_revenue_based_coef_step) = dill.load(f)

    cali_years = 8

    ### Load observation data
    data = pd.read_csv(join(wd, "Inputs", "Data_SD6_2012_2022_ci_test2011.csv"), index_col=["year"])
    #prec_avg = pd.read_csv(join(wd, "Inputs", "prec_avg_2012_2022.csv"), index_col=[0]).iloc[1:, :]
    # Normalize GW_st withdrawal to [0, 1] according to obv
    data["GW_st"] = (data["GW_st"]-17.5577) / (18.2131-17.5577)
    data["withdrawal"] = (data["withdrawal"]-1310.6749) / (3432.4528-1310.6749)
    data = data.iloc[1:cali_years, :]

    crop_options = ["corn", "sorghum", "soybeans", "wheat", "fallow"]

    ### Read PSO variables
    i_iter = kwargs.get("i_iter")
    i_particle = kwargs.get("i_particle")

    ### Setup parameters

    # for fid in fields_dict:
    #     fields_dict[fid]["water_yield_curves"]["others"] = [x[0], x[1], x[2], x[3], x[4], 0.1186]
    # for yr in crop_price_step["finance"]:
    #     crop_price_step["finance"][yr]["others"] *= x[5]

    pars = {
        'perceived_risk': x[0],
        'forecast_trust': x[1],
        'sa_thre': x[2],
        'un_thre': x[3]
        }

    rmse_list = []
    for seed in seeds:
        try:
            m = SD6Model_1f1w_ci(
                pars=pars,
                crop_options=crop_options,
                aquifers_dict=aquifers_dict,
                fields_dict=fields_dict,
                wells_dict=wells_dict,
                finances_dict=finances_dict,
                behaviors_dict=behaviors_dict,
                prec_aw_step=prec_aw_step,
                init_year=2011,
                end_year=2022,
                lema_options=(True, 'wr_LEMA_5yr', 2013),
                fix_state=None,
                show_step=True,
                seed=seed,
                activate_ci=True,
                # kwargs
                gurobi_dict={"LogToConsole": 0, "NonConvex": 2, "Presolve": -1},
                crop_price_step=crop_price_step,
                harvest_price_step=harvest_price_step,
                projected_price_step=projected_price_step,
                aph_revenue_based_coef_step=aph_revenue_based_coef_step
                )

            m.particles = x
            for i in range(cali_years):  # use first ten years to calibrate
                m.step()
            m.end() # despose gurobi env

            # Output df
            df_sys, _, _ = m.get_dfs(m)

            # Normalize GW_st withdrawal to [0, 1] according to obv (i.e., data)
            df_sys["GW_st"] = (df_sys["GW_st"]-17.5577) / (18.2131-17.5577)
            df_sys["withdrawal"] = (df_sys["withdrawal"]-1310.6749) / (3432.4528-1310.6749)

            # Calculate metrices
            metrices = m.get_metrices(df_sys.iloc[1:,:], data, targets=['GW_st', 'withdrawal']
                + crop_options, indicators_list=["rmse"])

            # Calculate obj
            rmse_sys = metrices.loc[['GW_st', 'withdrawal'], "rmse"].mean()
            rmse_crop = metrices.loc[crop_options, "rmse"].mean()
            rmse = (rmse_sys + rmse_crop)/2
            # save the model
            m.rmse = rmse
            sys.setrecursionlimit(10000)  # Set to a higher value
            with open(join(exp_dir, f"{round(rmse,5)*100000}_it{i_iter}_ip{i_particle}_s{seed}.pkl"), "wb") as f:
                dill.dump(m, f)
            #del m
        except Exception as e:
            error_message = str(e)
            print(f"An error occurred: {error_message}")
            rmse = 1000  # in case any error occur
        #Add to the list
        rmse_list.append(rmse)

    #cost = sum(rmse_list)/len(rmse_list)
    cost = min(rmse_list)
    seed = seeds[int(np.argmin(rmse_list))]
    with open(join(exp_dir, f"{round(cost,3)*1000}_it{i_iter}_ip{i_particle}_s{seed}.txt"), "w") as f:
        f.write(f"it{i_iter}_ip{i_particle}_s{seed}\nRMSE: {cost}\nx: {x}")

    return cost
#%% Setup PSO
# Info
n_particles = 3 #24
dimensions = 4
options = {'c1': 0.5, 'c2': 0.5, 'w':0.8} # hyperparameters {'c1', 'c2', 'w', 'k', 'p'}

# Manually defined starting point for the first particle
manual_starting_point = [0.588505857, 0.538740883, 0.161558484, 0.353470650] 

# Bounds
lowerbounds = [0, 0, 0, 0]    #[0]*4
upperbounds = [1, 1, 1, 1]    #[1]*4
# lowerbounds = [0.5, 0.5, 0, 0]    #[0]*4
# upperbounds = [1, 1, 0.5, 0.5]    #[1]*4

rngen = np.random.default_rng(seed=12345)
init_pos = rngen.uniform(0,1,(n_particles, dimensions))
for i in range(dimensions):
    init_pos[:, i] = init_pos[:, i]*(upperbounds[i]-lowerbounds[i]) + lowerbounds[i]
    
# Set the first particle to the manually defined starting point
init_pos[0] = manual_starting_point 
# Ensure that all points are within the bounds 
init_pos = np.clip(init_pos, lowerbounds, upperbounds)
#%%
# Initialize PSO
optimizer = GlobalBestPSO(
    n_particles=n_particles, dimensions=dimensions, options=options,
    bounds=(lowerbounds, upperbounds), init_pos=init_pos,
    wd=exp_dir)

# N = 5
# rng = np.random.default_rng(12345)
# seeds = [int(rng.integers(low=0, high=999999)) for _ in range(N)]
seeds = [3, 56, 67] #seeds = [3, 56, 67]
# Run PSO
timer = TimeRecorder()
cost, pos = optimizer.optimize(obj_func, iters=10, n_processes=2, verbose=60,
                               seeds=seeds, exp_dir=exp_dir)

print("\a")
elapsed_time = timer.get_elapsed_time()
print(elapsed_time)
