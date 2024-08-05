# %%
import os
import sys
import dill
import pandas as pd
import seaborn
import datetime

from py_champ.models.sd6_model_1f1w import SD6Model4SingleFieldAndWell

from plot_1f1w import (plot_GW_st,
                       plot_withdrawal,
                       plot_crop_ratio,
                       plot_rainfed,
                       plot_aquifer,
                       plot_field,
                       plot_well,
                       plot_finance,
                       plot_behavior)

# Define the working directory

# ADD FUNCTION TO GET ABSOLUTE PATH AND DEFINED IT!!

# Malena PC ->
#wd = r"D:\Malena\CHAMP\PyCHAMP\code_20240704\PyCHAMP\examples\Heterogeneity"
# Malena Laptop ->
#wd = r"C:\Users\m154o020\CHAMP\PyCHAMP\Summer2024\code_20240705\PyCHAMP\examples\Heterogeneity"
# Michelle Laptop 
wd = r"/Users/michellenguyen/Downloads/PyCHAMP/examples/Heterogeneity"

# Set this to True if you want to load from Outputs directory
load_from_outputs = False

# Add file paths dynamically
def add_file(file_name, alias):
    setattr(paths, alias, os.path.join(wd, file_name))

paths = type("Paths", (), {})

# Add inputs files
init_year = 2011
add_file(f"Inputs_SD6_{init_year+1}_2022.pkl", "input_pkl")
add_file(f"prec_avg_{init_year}_2022.csv", "prec_avg")
add_file(f"Data_SD6_{init_year+1}_2022.csv", "sd6_data")
add_file("calibrated_parameters.txt", "cali_x")

# Load data
prec_avg = pd.read_csv(paths.prec_avg, index_col=[0]).iloc[1:, :]
sd6_data = pd.read_csv(paths.sd6_data, index_col=["year"])
crop_options = ["corn", "others"]

# Run the model if not loading from outputs
if not load_from_outputs:
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

    # Run the model
    m = SD6Model4SingleFieldAndWell(
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

# =============================================================================
# Analyze results
# =============================================================================
    output_dir = os.path.join(wd, f"Outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    data = sd6_data
    df_sys, df_agt = m.get_dfs(m)
    metrices = m.get_metrices(df_sys, data) # same length

    # Save df_sys and df_agt as CSV files
    df_sys.to_csv(os.path.join(output_dir, f"df_sys_{timestamp}.csv"), index=True)
    df_agt.to_csv(os.path.join(output_dir, f"df_agt_{timestamp}.csv"), index=True)
    metrices.to_csv(os.path.join(output_dir, f"metrices_{timestamp}.csv"), index=True)

else:
    # Load latest CSV files from Outputs directory
    output_dir = os.path.join(wd, "Outputs")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    df_sys_filename = os.path.join(output_dir, "df_sys_20240718_204825.csv")
    df_agt_filename = os.path.join(output_dir, "df_agt_20240718_204825.csv")
    metrices_filename = os.path.join(output_dir, "metrices_20240718_204825.csv")

    df_sys = pd.read_csv(os.path.join(output_dir, df_sys_filename), index_col=0)
    df_agt = pd.read_csv(os.path.join(output_dir, df_agt_filename), index_col=0)
    metrices = pd.read_csv(os.path.join(output_dir, metrices_filename), index_col=0)
    data = sd6_data

    if df_sys.index.duplicated().any():
        df_sys = df_sys[~df_sys.index.duplicated(keep='first')]

# =============================================================================
# Plot results
# =============================================================================
plot_GW_st(df_sys.reindex(data.index),
             data,
             metrices,
             prec_avg,
             crop_options,
             savefig=os.path.join(output_dir, f"GW_st_{timestamp}.png"))

plot_withdrawal(df_sys.reindex(data.index),
                 data,
                 metrices,
                 prec_avg,
                 crop_options,
                 savefig=os.path.join(output_dir, f"withdrawal_{timestamp}.png"))

plot_crop_ratio(df_sys.reindex(data.index),
                 data,
                 metrices,
                 prec_avg,
                 crop_options,
                 savefig=os.path.join(output_dir, f"crop_ratio_{timestamp}.png"))

plot_rainfed(df_sys.reindex(data.index),
                 data,
                 metrices,
                 prec_avg,
                 crop_options,
                 savefig=os.path.join(output_dir, f"rainfed_{timestamp}.png"))

plot_aquifer(df_sys.reindex(data.index),
                 data,
                 metrices,
                 prec_avg,
                 crop_options,
                 savefig=os.path.join(output_dir, f"aquifer_{timestamp}.png"))

plot_field(df_sys.reindex(data.index),
                 data,
                 metrices,
                 prec_avg,
                 crop_options,
                 savefig=os.path.join(output_dir, f"field_{timestamp}.png"))


# FIX PLOTS BELOW
plot_well(df_sys.reindex(data.index),
                 data,
                 metrices,
                 prec_avg,
                 crop_options,
                 savefig=os.path.join(output_dir, f"well_{timestamp}.png"))

plot_finance(df_sys.reindex(data.index),
                 data,
                 metrices,
                 prec_avg,
                 crop_options,
                 savefig=os.path.join(output_dir, f"finance_{timestamp}.png"))

plot_behavior(df_sys.reindex(data.index),
                 data,
                 metrices,
                 prec_avg,
                 crop_options,
                 savefig=os.path.join(output_dir, f"behavior_{timestamp}.png"))