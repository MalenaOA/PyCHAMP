# %%
import os
import sys
import dill
import pandas as pd

from py_champ.models.sd6_model_aquacrop import SD6ModelAquacrop

# Define the working directory
#wd = r"D:\Malena\CHAMP\PyCHAMP\code_20240704\PyCHAMP\examples\Heterogeneity"
wd = r"/Users/michellenguyen/Downloads/PyCHAMP/examples/Heterogeneity"
# Add the 'code' directory to sys.path if not already present
if wd not in sys.path:
    sys.path.append(os.path.join(wd, "code"))

# Define a function to add file paths dynamically
def add_file(file_name, alias):
    setattr(paths, alias, os.path.join(wd, file_name))

# Initialize paths as an empty object
paths = type("Paths", (), {})

# Add file paths using the add_file function
init_year = 2011
add_file(f"Inputs_SD6_{init_year+1}_2022.pkl", "input_pkl")
add_file(f"prec_avg_{init_year}_2022.csv", "prec_avg")
add_file(f"Data_SD6_{init_year+1}_2022.csv", "sd6_data")
add_file("calibrated_parameters.txt", "cali_x")

# Load inputs
with open(paths.input_pkl, "rb") as f:
    (
        aquifers_dict,
        fields_dict,
        wells_dict,
        finances_dict,
        behaviors_dict,
        prec_aw_step,
        crop_price_step,
    ) = dill.load(f)

# Load data
prec_avg = pd.read_csv(paths.prec_avg, index_col=[0]).iloc[1:, :]
sd6_data = pd.read_csv(paths.sd6_data, index_col=["year"])

# General model settings
crop_options = ["corn", "others"]

# Function to load parameters from the text file
def load_parameters(file_path):
    x = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # print(lines)
        for line in lines:
            if line.startswith("x:"):
                x_str = line.split('[', 1)[1].split(']', 1)[0].strip()
                # print(x_str)
                x = list(map(float, x_str.split()))
    return x


# Load parameters from the calibrated_parameters.txt file
x = load_parameters(paths.cali_x)

for fid in fields_dict:
    fields_dict[fid]["water_yield_curves"]["others"] = [
        x[0],
        x[1],
        x[2],
        x[3],
        x[4],
        0.1186,
    ]
for yr in crop_price_step["finance"]:
    crop_price_step["finance"][yr]["others"] *= x[5]

pars = {
    "perceived_risk": x[6],
    "forecast_trust": x[7],
    "sa_thre": x[8],
    "un_thre": x[9],
}

# %%
# Run the model
m = SD6ModelAquacrop(
    pars=pars,
    crop_options=crop_options,
    prec_aw_step=prec_aw_step,
    aquifers_dict=aquifers_dict,
    fields_dict=fields_dict,
    wells_dict=wells_dict,
    finances_dict=finances_dict,
    behaviors_dict=behaviors_dict,
    crop_price_step=crop_price_step,
    init_year=init_year,
    end_year=2022,
    lema_options=(True, "wr_LEMA_5yr", 2013),
    show_step=True,
    seed=67,
)

for i in range(11):
    m.step()

m.end()

# # %%
# # Analyze the results

data = sd6_data
df_sys, df_agt = m.get_dfs(m)
#
# visual = SD6Visual()
# visual.add_sd6_plotting_info(df_sys=df_sys, sd6_data=sd6_data, prec_avg=prec_avg)
# visual.plot_timeseries()
# visual.plot_crop_ratio()
#
# # %%

# read outputs for attributes related to different agent types
df_farmers, df_fields, df_wells, df_aquifers = SD6ModelAquacrop.get_dfs(m)

# read system level outputs. For e.g., ratios of crop types, irrigation technology, rainfed or irrigated field for the duration of the simulation
#df_sys = SD6ModelAquacrop.get_df_sys(m, df_farmers, df_fields, df_wells, df_aquifers)
metrices = m.get_metrices(df_sys, data) # same length
print(df_fields)

import seaborn

from plot_1f1w import (plot_cali_gwrc, plots_simulation, reg_prec_withdrawal)

# Plot results
plot_cali_gwrc(df_sys.reindex(data.index), data, metrices, prec_avg, stochastic=[], savefig=None)

plots_simulation(df_sys.reindex(data.index), data, metrices, prec_avg, savefig=None)

reg_prec_withdrawal(prec_avg, df_sys.reindex(data.index), df_sys_nolema=None, data=data,
                    df_sys_list=None, df_sys_nolema_list=None, dot_labels=True, obv_dots=False, savefig=None)