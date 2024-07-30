import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import pandas as pd
from copy import deepcopy
from adjustText import adjust_text

# Global font properties
matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Helvetica')
matplotlib.rc('text', usetex='false')
matplotlib.rc('font', size=9)

# Colors
col = {"dry_year": "#C1C0C0",
       "GW_st": "#377EB8",
       "withdrawal": "#4472C4",
       "rainfed": "#4DAF4A",
       "irrigated": "#4472C4",
       "crop": "#4DAF4A",
       "corn": "#FDC829",
       "sorghum": "#8D464C",
       "soybeans": "#216D00",
       "wheat": "#A36D00",
       "fallow": "#3E3E3E",
       "others": "#8D464C",
       "profit": "#FFC000",
       "energy": "#E41A1C",
       "Imitation": "#13BDCF",
       "Social comparison": "#E27BC3",
       "Repetition": "#FF7F00",
       "Deliberation": "#984EA3",
       "k": "#000000"}

f_s = {"single": (7, 3.5)}
lab_s = 8
tit_s = 11
dpi = 1000

# ===========================================================
# Wet Dry background
def plot_wet_dry_bc(ax, prec_avg):
    if prec_avg is not None:
        prec_avg["year"] = prec_avg.index
        mean_rainfall = prec_avg["annual"].mean()
        for index, row in prec_avg.iterrows():
            if row['annual'] <= mean_rainfall:
                ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color=col["dry_year"], alpha=0.5, lw=0)
    ax.axvline(2013, color=col["k"], ls=":", lw=1)
    # ax.axvline(2018, color=col["k"], ls=":", lw=1)

# ===========================================================
# Plot Settings
def plot_settings(fig, ax, y_label=None, title=None, suptitle=None, sup_color=None):
    ax.set_xlim([2012, 2022])
    ax.set_xlabel("Years")
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title, fontsize=tit_s, fontweight='bold')
    if suptitle and sup_color:
        ax.text(0.0, 1.1, suptitle, transform=ax.transAxes, color=sup_color, fontweight='bold', ha='right', va='bottom', fontsize=tit_s)
    ax.legend(loc="best", bbox_to_anchor=(1.01, 0.7), ncols=1, frameon=False, fontsize=lab_s)
    # fig.tight_layout(pad=0.5, h_pad=1.0)

# ===========================================================
def plot_results(df_sys=None,
                 prec_avg=None,
                 crop_options=None,
                 background=None,
                 savefig=None,
                 combined=None):
    total_subplots = 8
    if combined:
        figsize = (f_s['single'][0], f_s['single'][1] * total_subplots)
        fig, axes = plt.subplots(ncols=1, nrows=total_subplots, figsize=figsize)
    else:
        fig, axes = None, [plt.subplots(figsize=f_s['single'])[1] for _ in range(total_subplots)]

    x = list(df_sys.index)

    if background:
        for ax in axes:
            plot_wet_dry_bc(ax, prec_avg)

    # Aquifer - irrigation volume total and by crops
    ax = axes[0]
    c_hex_list = [col[crop] for crop in crop_options]
    crop_irr_vol_cols = [f"{crop}_irr_vol" for crop in crop_options]
    if df_sys is not None:
        for i, crop_col in enumerate(crop_irr_vol_cols):
            ax.plot(x, df_sys[crop_col], color=c_hex_list[i], label=crop_options[i].capitalize())
        ax.plot(x, df_sys['total_irr_vol'], color=col["k"], label="Total")
    plot_settings(fig,
                  ax,
                  y_label="($1E^6$ $m^3$)",
                  title="Irrigation Volume",
                  suptitle="AQUIFER",
                  sup_color=col["withdrawal"])

    # Field - Field types
    ax = axes[1]
    if df_sys is not None:
        ax.plot(x, df_sys['rainfed'], color=col['rainfed'], label='Rainfed')
        ax.plot(x, df_sys['irrigated'], color=col['irrigated'], label='Irrigated')
    plot_settings(fig,
                  ax,
                  y_label="Ratio",
                  title="Field Types",
                  suptitle="FIELD",
                  sup_color=col["crop"])

    # Field - Crop yields
    ax = axes[2]
    c_hex_list = [col[crop] for crop in crop_options]
    for i, crop in enumerate(crop_options):
        if df_sys is not None:
            yield_col = f"{crop}_yield"
            ax.plot(x, df_sys[yield_col], color=c_hex_list[i], label=crop.capitalize())
    plot_settings(fig,
                  ax,
                  y_label="($1E^4$ bu)",
                  title="Crop Yield",
                  suptitle="FIELD",
                  sup_color=col["crop"])

    # Well - Total energy
    ax = axes[3]
    if df_sys is not None:
        ax.plot(x, df_sys['total_energy'], color=col["energy"], label="Sim")
    plot_settings(fig,
                  ax,
                  y_label="(PJ)",
                  title="Energy Use",
                  suptitle="WELL",
                  sup_color=col["energy"])

    # Well - Energy per irrigation depth
    ax = axes[4]
    if df_sys is not None:
        ax.plot(x, df_sys['energy_irr_depth'], color=col["energy"], label="Sim")
    plot_settings(fig,
                  ax,
                  y_label="(PJ/cm)",
                  title="Energy Use per Irrigation Depth",
                  suptitle="ENERGY",
                  sup_color=col["energy"])

    # Finance - crop profits and total profit
    ax = axes[5]
    c_hex_list = [col[crop] for crop in crop_options]
    crop_profit_cols = [f"{crop}_profit" for crop in crop_options]
    if df_sys is not None:
        for i, crop_col in enumerate(crop_profit_cols):
            ax.plot(x, df_sys[crop_col], color=c_hex_list[i], label=crop_options[i].capitalize())
        ax.plot(x, df_sys['total_profit'], color=col["k"], label="Total")
    plot_settings(fig,
                  ax,
                  y_label="($1E^4$ $)",
                  title="Profits",
                  suptitle="FINANCE",
                  sup_color=col["profit"])

    # Finance - profit per irrigation depth
    ax = axes[6]
    if df_sys is not None:
        ax.plot(x, df_sys['profit_irr_depth'], color=col["profit"], label="Sim")
    plot_settings(fig,
                  ax,
                  y_label="($1E^4$ $/cm)",
                  title="Profit per Irrigation Depth",
                  suptitle="FINANCE",
                  sup_color=col["profit"])

    # CONSUMAT State ratio
    ax = axes[7]
    states = ["Imitation", "Social comparison", "Repetition", "Deliberation"]
    c_hex_list = [col[state] for state in states]
    if df_sys is not None:
        dff = df_sys[states]
        num_agt = sum(dff.iloc[0, :])
        dff = (dff / num_agt)*100
        for i, state in enumerate(states):
            ax.plot(x, dff[state], color=c_hex_list[i], label=state)
    plot_settings(fig,
                  ax,
                  y_label="(% Farmers)",
                  title="CONSUMAT States",
                  suptitle="BEHAVIOR",
                  sup_color=col["Repetition"])

    if savefig is not None:
        if combined:
            fig.tight_layout(pad=0.5, h_pad=1.0)
            plt.savefig(savefig, dpi=dpi)
        else:
            for i, ax in enumerate(axes):
                fig = ax.get_figure()
                fig.tight_layout(pad=0.5)
                plt.savefig(savefig, dpi=dpi)

    if combined:
        plt.show()
    else:
        for ax in axes:
            fig = ax.get_figure()
            fig.show()