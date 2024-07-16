import os
import sys
import dill
import numpy as np
import pandas as pd
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from geopy.distance import geodesic

wd = rf"C:/Users/{os.getlogin()}/Documents/GitHub/EnvHeteroImpactGW"
if wd not in sys.path:
    sys.path.append(os.path.join(wd, "code"))

from utils import ProjPaths

# Add file paths
paths = ProjPaths(wd)
paths.add_file("SD6_grid_info.csv", "data")
paths.add_file("KFMA_crop_income.csv", "data")
paths.add_file("prec_D_cm_1990_2023.csv", "data")
paths.add_file("PDIV_WaterUse_1990_2022.csv", "data")

init = 2011
start = init + 1
end = 2022
crop_options = ["corn", "others"]
growing_seasons = {
    "corn": ["5/1", "10/3"],
    "others": ["5/1", "10/3"],
}
nprng = np.random.default_rng(12345)

# SD6 grid data
sd6_grid_info = pd.read_csv(paths.SD6_grid_info)
# Select grids based on other crop type frequency
seleted_SD6_grids = sd6_grid_info[sd6_grid_info["other_freq"] <= 3].reset_index()
fnum = seleted_SD6_grids.shape[0]
seleted_SD6_grids["fid"] = [f"f{i+1}" for i in range(fnum)]  # assign field id
seleted_SD6_grids["wid"] = [f"w{i+1}" for i in range(fnum)]  # assign well id
seleted_SD6_grids["bid"] = [f"b{i+1}" for i in range(fnum)]  # assign behavior id
seleted_SD6_grids["aqid"] = "sd6"  # assign acquifer id

# =============================================================================
# Crop price (KFMA) Inputs
# =============================================================================
# We calculated a weighted mean using Ymax from EMS paper.
kfma_income = pd.read_csv(paths.KFMA_crop_income, index_col="Year")
kfma_income["others"] = (
    kfma_income["sorghum"] * 194.0593
    + kfma_income["soybeans"] * 146.3238
    + kfma_income["wheat"] * 141.1518
) / (194.0593 + 146.3238 + 141.1518)
kfma_income.plot(xlabel="Year", ylabel="$/bu", xlim=[2008, 2022]).legend(
    ncol=2, frameon=False
)
crop_price_step = {
    "finance": kfma_income.loc[init:end, crop_options].round(3).T.to_dict()
}

# =============================================================================
# Prec_aw Inputs
# =============================================================================
# Climate (gridMET)
selected_gridmet_grids = [
    "grid37",
    "grid9",
    "grid26",
    "grid27",
    "grid17",
    "grid29",
    "grid10",
    "grid7",
    "grid4",
    "grid5",
    "grid34",
    "grid8",
    "grid24",
    "grid38",
    "grid23",
    "grid19",
    "grid14",
    "grid43",
    "grid12",
    "grid42",
    "grid28",
    "grid18",
    "grid32",
    "grid21",
    "grid22",
    "grid33",
    "grid13",
]
prec = pd.read_csv(paths.prec_D_cm_1990_2023, parse_dates=[0], index_col=[0])
prec = prec.loc[str(init) : str(end + 1), selected_gridmet_grids]


def cal_prec_aw(prec, growing_seasons, start, end, crop_options):
    if prec.index[-1].year < end + 1:
        raise ValueError("The data needs to be one year longer then the end year.")
    dfs = {}
    for crop in crop_options:
        if crop == "fallow":
            d = prec.copy()
            df = d.resample("YS").sum()[str(start) : str(end)]
            df.index = df.index.year
            df.loc[:, :] = 0
        else:
            periods = growing_seasons[crop]
            s = pd.to_datetime(periods[0], format="%m/%d")
            e = pd.to_datetime(periods[1], format="%m/%d")
            m = s.month - 1
            s = s - relativedelta(months=m)
            e = e - relativedelta(months=m)
            d = prec.copy()
            d.index = [i - relativedelta(months=m) for i in list(d.index)]

            mask = (
                pd.Series(d.index.strftime("%m%d").astype(int))
                .between(int(s.strftime("%m%d")), int(e.strftime("%m%d")))
                .to_list()
            )
            df = d[mask].resample("YS").sum()[str(start) : str(end)]
            df.index = df.index.year
        dfs[crop] = df.round(2)
    return dfs


# Annual prec
df_prec_avg = pd.DataFrame()
df_prec = prec.resample("YS").sum()[str(init) : str(end)].round(2)
df_prec.index = df_prec.index.year
df_prec_avg["annual"] = df_prec.mean(axis=1)

# Annual available precipitaion during growing_seasons
df_prec_aw = cal_prec_aw(prec, growing_seasons, init, end, crop_options)
for crop, prec_aw in df_prec_aw.items():
    df_prec_avg[crop] = prec_aw.mean(axis=1)
df_prec_avg.to_csv(os.path.join(paths.data, rf"prec_avg_{init}_{end}.csv"))

# Calculate trancated normal distribution parameters for calculating perceived
# risks using scipy (a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale)
# We still using 2007 to 2022 in case data is insufficient
df_prec_aw_ = cal_prec_aw(prec, growing_seasons, 2007, end, crop_options)
truncated_normal = pd.DataFrame()
for crop in crop_options:
    if crop == "fallow":
        truncated_normal[crop] = None
        continue
    df_prec_aw_crop = df_prec_aw_[crop]
    loc = df_prec_aw_crop.mean().to_frame()
    scale = df_prec_aw_crop.std().to_frame()
    # assume 2 sigma on each side of loc 95% interval
    a = -pd.DataFrame({0: [2] * len(loc)}, index=loc.index)
    b = -a
    # check whether below 0
    dd = truncnorm.ppf(0, a, b, loc, scale)
    for i in np.where(dd < 0)[0]:
        a.iloc[i, 0] = (0 - loc.iloc[i, 0]) / scale.iloc[i, 0]
    pars = (
        pd.concat([a, b, loc, scale])
        .stack()
        .groupby(level=[0, 1])
        .apply(tuple)
        .unstack()
    )
    truncated_normal[crop] = pars.sort_index()
truncated_normal = truncated_normal.T.to_dict()

# Form prec_aw_step
prec_aw_step = {}
for gridmet_id in selected_gridmet_grids:
    df = pd.concat([df_prec_aw[crop][[gridmet_id]] for crop in crop_options], axis=1)
    df.columns = crop_options
    prec_aw_step[gridmet_id] = df.T.to_dict()

# =============================================================================
# Aquifer Inputs
# =============================================================================
# Let's use st and set initial dwl to -0.4
# 2002–2016 data (ΔWL [m] = 0.767 – 0.0376 Q [1e6 m3], p < 0.0001)
# = ΔWL [m] = 0.767 – 0.000376 Q [1e4 m3 = m-ha]
aquifers_dict = {
    "sd6": {
        "aq_a": 0.0003310,  # old 0.000376
        "aq_b": 0.6286,  # old 0.767
        "init": {"st": sd6_grid_info[f"st_m_{init}"].mean(), "dwl": -0.4},
    }
}

# =============================================================================
# Field Inputs
# =============================================================================
fields_dict = {}
seleted_SD6_grids.index = seleted_SD6_grids["fid"]
for _, fid in enumerate(seleted_SD6_grids["fid"]):
    init_crop = seleted_SD6_grids.loc[fid, f"Crop{init}"]
    if init_crop != "corn":
        init_crop = "others"
    prec_aw_id = seleted_SD6_grids.loc[fid, "gridmet_id"]
    fields_dict[fid] = {
        "field_area": 50.0,  # [ha] field size.
        "water_yield_curves": {
            "corn": [463.3923, 77.7756, -3.3901, 6.0872, -1.7325, 0.1319],
            # parameters for others are subjected to be calibrated
            "others": [160.5116, 66.1485, -2.1833, 3.8169, -0.6708, 0.1186],
        },
        "prec_aw_id": prec_aw_id,
        "init": {
            "crop": init_crop,
            "field_type": "optimize",  # "optimize" or "irrigated" or "rainfed"
        },
        # Additional info
        "truncated_normal_pars": truncated_normal[prec_aw_id],
        "irr_freq": seleted_SD6_grids.loc[fid, "irr_ratio"],
        "lat": seleted_SD6_grids.loc[fid, "lat"],
        "lon": seleted_SD6_grids.loc[fid, "lon"],
        "y": seleted_SD6_grids.loc[fid, "Y"],
        "x": seleted_SD6_grids.loc[fid, "X"],
    }

# =============================================================================
# Well Inputs
# =============================================================================
# Assign the nearest well data that we have
wells_dict = {}
seleted_SD6_grids.index = seleted_SD6_grids["wid"]
seleted_SD6_grids["well_st"] = (
    seleted_SD6_grids.loc[:, f"wl_ele_m_{init}"]
    - seleted_SD6_grids.loc[:, "well_depth_ele"]
)
pumping_days = 90
r = 0.2032  # [m] = 8 inches
eff_well = 0.5

for _, wid in enumerate(seleted_SD6_grids["wid"]):
    tr = seleted_SD6_grids.loc[wid, "well_st"] * seleted_SD6_grids.loc[wid, "well_k"]
    wells_dict[wid] = {
        "rho": 1000.0,  # [kg/m3]
        "g": 9.8016,  # [m/s2]
        "eff_pump": 0.77,
        "aquifer_id": seleted_SD6_grids.loc[wid, "aqid"],
        "pumping_capacity": None,
        "init": {
            "B": 1
            / (4 * np.pi * tr * eff_well)
            * (
                -0.5772
                - np.log(
                    r**2
                    * seleted_SD6_grids.loc[wid, "well_sy"]
                    / (4 * tr * pumping_days)
                )
            ),
            "l_wt": seleted_SD6_grids.loc[wid, f"wl_depth_m_{init}"],
            "pumping_days": pumping_days,
        },
    }


# =============================================================================
# Finance Inputs
# =============================================================================
finances_dict = {
    "finance": {
        "energy_price": 2777.777778,  # [1e4$/PJ] $0.10/kWh = $ 2777.777778 1e4/PJ
        "crop_price": {
            "corn": crop_price_step["finance"][init]["corn"],
            # There is a ratio parameter for others that is subjected to be calibrated.
            "others": crop_price_step["finance"][init]["others"],
        },
        "crop_cost": {
            "corn": 0,  # $/bu
            "others": 0,  # $/bu
        },
        "irr_tech_operational_cost": {"center pivot LEPA": 1.876},  # [1e4$]
    }
}


# =============================================================================
# Behavior Inputs
# =============================================================================
# Calculate neighbors by distance
def get_neighbors_in_buffer_circle(df, id_col, radius_km=1):
    """df = sd6_grid_info[["gid", "lon", "lat"]]"""
    ids = df[id_col].values
    lons = df["lon"].values
    lats = df["lat"].values
    neighbors = {}
    for id, lon, lat in tqdm(zip(ids, lons, lats)):
        # The default is the WGS-84 ellipsoid
        dis = [geodesic((lat, lon), (y, x)).km for x, y in zip(lons, lats)]
        neighbors[id] = list(ids[np.where(np.array(dis) <= radius_km)[0]])
        neighbors[id].remove(id)
        if neighbors[id] == []:
            second_smallest_dis = sorted(set(dis))[1]
            neighbors[id] = list(ids[np.where(np.array(dis) == second_smallest_dis)[0]])
            print(
                f"\n{id}: Select the closest but out of the radius agent as its neighbor."
            )
    return neighbors


behaviors_dict = {}
in2cm = 2.54
seleted_SD6_grids.index = seleted_SD6_grids["bid"]
df_behaviors = seleted_SD6_grids.copy()
neighbors = get_neighbors_in_buffer_circle(df=df_behaviors, id_col="bid", radius_km=1)

for _, bid in enumerate(df_behaviors["bid"]):
    behaviors_dict[bid] = {
        "field_ids": [df_behaviors.loc[bid, "fid"]],  # for single field & well only
        "well_ids": [df_behaviors.loc[bid, "wid"]],  # for single field & well only
        "finance_id": "finance",
        "behavior_ids_in_network": neighbors[bid],
        "decision_making": {
            "target": "profit",
            "horizon": 5,
            "n_dwl": 5,
            "keep_gp_model": False,
            "keep_gp_output": False,
            "display_summary": False,
            "display_report": False,
        },
        "water_rights": {
            "wr_yr": {
                "wr_depth": 24 * in2cm,
                "time_window": 1,
                "remaining_tw": None,
                "remaining_wr": None,
                "tail_method": "proportion",
                "status": True,
            },
            "wr_LEMA_5yr": {
                "wr_depth": 11 * 5 * in2cm,  # 139.7 = 27.94 * 5 cm
                "time_window": 5,
                "remaining_tw": None,
                "remaining_wr": None,
                "tail_method": "proportion",
                "status": False,
            },
        },
        "consumat": {
            # [0-1] Sensitivity factor for the "satisfication" calculation.
            "alpha": {"profit": 1},
            # Normalize "need" for "satisfication" calculation.
            "scale": {"profit": 0.23 * 50},  # Use corn 1e4$*bu*ha
        },
    }

# Output
inputs = (
    aquifers_dict,
    fields_dict,
    wells_dict,
    finances_dict,
    behaviors_dict,
    prec_aw_step,
    crop_price_step,
)
with open(os.path.join(paths.inputs, rf"Inputs_SD6_{start}_{end}.pkl"), "wb") as f:
    dill.dump(inputs, f)

# %%
# =============================================================================
# Calibration data
# =============================================================================
# WIMAS
df_pdiv_water_use = pd.read_csv(paths.PDIV_WaterUse_1990_2022)
# Aquifer saturated thickness
df_st = (
    sd6_grid_info[[f"st_m_{y}" for y in range(start, end + 1)]]
    .mean()
    .to_frame(name="GW_st")
)
df_st.index = np.arange(start, end + 1)

# Withdrawal
withdrawal = df_pdiv_water_use[[f"AF_USED_IRR_{y}" for y in range(start, end + 1)]]
withdrawal = withdrawal.sum().to_frame(name="withdrawal") * 0.123348185532  # AF to m-ha
withdrawal.index = [y for y in range(start, end + 1)]

# Crop ratio
df = seleted_SD6_grids[[f"Crop{y}" for y in range(start, end + 1)]]
df[df != "corn"] = "others"
df["count"] = 1
df_crop = pd.concat(
    [
        df[[f"Crop{y}", "count"]].groupby(f"Crop{y}").count().reindex(crop_options)
        for y in range(start, end + 1)
    ],
    axis=1,
).T
df_crop.index = np.arange(start, end + 1)
df_crop_ratio = df_crop / df_crop.sum(axis=1).values.reshape((-1, 1))

# Field type ratio
df = seleted_SD6_grids[[f"Irr{y}" for y in range(start, 2020 + 1)]] - 1
df[df == -1] += 2
df_rainfed_ratio = (df.sum(axis=0) / df.shape[0]).to_frame(name="rainfed")
df_rainfed_ratio.index = np.arange(start, 2020 + 1)
df_rainfed_ratio = df_rainfed_ratio.reindex(np.arange(start, end + 1))

# Output
data = pd.concat([df_st, withdrawal, df_crop_ratio, df_rainfed_ratio], axis=1)
data.index = data.index.set_names("year")
data.to_csv(os.path.join(paths.data, rf"Data_SD6_{start}_{end}.csv"))
