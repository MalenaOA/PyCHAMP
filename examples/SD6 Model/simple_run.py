'''
terminal:
python.exe -m pip install --upgrade pip
pip install git+https://github.com/MalenaOA/PyCHAMP.git
'''
import os

# wd = r"C:\Users\CL\OneDrive\VT\Proj_DIESE\Code"
# Malena PC ->
#wd = r"D:\Malena\CHAMP\PyCHAMP\code_20240704\PyCHAMP"
wd = r"/Users/michellenguyen/Downloads/PyCHAMP"
# Malena Laptop ->
# wd = r"C:\Users\m154o020\CHAMP\PyCHAMP\Summer2024\code_20240705\PyCHAMP\examples\SD6 Model"
import sys

sys.setrecursionlimit(10000)
import dill

from py_champ.models.sd6_model import SD6Model

# %%
# =============================================================================
# Run simulation
# =============================================================================
# Load data
# Malena PC ->
#wd = r"D:\Malena\CHAMP\PyCHAMP\code_20240704\PyCHAMP\examples\SD6 Model"
# Malena Laptop ->
# wd = r"C:\Users\m154o020\CHAMP\PyCHAMP\Summer2024\code_20240705\PyCHAMP\examples\SD6 Model"
wd = r"/Users/michellenguyen/Downloads/PyCHAMP/examples/SD6 Model"
with open(os.path.join(wd, "Inputs_SD6.pkl"), "rb") as f:
    (
        aquifers_dict,
        fields_dict,
        wells_dict,
        finances_dict,
        behaviors_dict,
        prec_aw_step,
        crop_price_step,
        shared_config,
    ) = dill.load(f)

crop_options = ["corn", "sorghum", "soybeans", "wheat", "fallow"]
tech_options = ["center pivot LEPA"]
area_split = 1
seed = 3

pars = {
    "perceived_risk": 0.7539013390415119,
    "forecast_trust": 0.8032197483934305,
    "sa_thre": 0.14215821111637678,
    "un_thre": 0.0773514357873846,
}

m = SD6Model(
    pars=pars,
    crop_options=crop_options,
    tech_options=tech_options,
    area_split=area_split,
    aquifers_dict=aquifers_dict,
    fields_dict=fields_dict,
    wells_dict=wells_dict,
    finances_dict=finances_dict,
    behaviors_dict=behaviors_dict,
    prec_aw_step=prec_aw_step,
    init_year=2007,
    end_year=2022,
    lema_options=(True, "wr_LEMA_5yr", 2013),
    fix_state=None,
    show_step=True,
    seed=seed,
    shared_config=shared_config,
    # kwargs
    crop_price_step=crop_price_step,
)

for _i in range(15):
    m.step()

m.end()

# %%
# =============================================================================
# Analyze results
# =============================================================================
df_farmers, df_fields, df_wells, df_aquifers = SD6Model.get_dfs(m)
df_sys = SD6Model.get_df_sys(m, df_farmers, df_fields, df_wells, df_aquifers)

# df_sys["GW_st"].plot()
# df_sys["withdrawal"].plot()
# df_sys[["corn", "sorghum", "soybeans", "wheat", "fallow"]].plot()
# df_sys[["Imitation", "Social comparison", "Repetition", "Deliberation"]].plot()

import pandas as pd
data = pd.read_csv(os.path.join(wd, "Data_SD6.csv"), index_col=["year"])

prec_avg = pd.read_csv(os.path.join(wd, "Prec_avg.csv"), index_col=[0]).iloc[1:, :]

metrices = m.get_metrices(df_sys, data) # same length
print(df_fields)

import seaborn

from plot_EMS import (plot_cali_gwrc,
                      plot_crop_ratio,
                      reg_prec_withdrawal)

# Plot results
plot_cali_gwrc(df_sys.reindex(data.index),
               data,
               metrices,
               prec_avg,
               stochastic=[],
               savefig=None)

plot_crop_ratio(df_sys.reindex(data.index),
                data,
                metrices,
                prec_avg,
                savefig=None)

reg_prec_withdrawal(prec_avg,
                     df_sys.reindex(data.index),
                     df_sys_nolema=None,
                     data=data,
                     df_sys_list=None,
                     df_sys_nolema_list=None,
                     dot_labels=True,
                     obv_dots=False,
                     savefig=None)