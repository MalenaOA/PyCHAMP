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
       "k": "#000000"
}

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
def plot_settings(fig, ax, y_label=None, title=None, suptitle=None):
    ax.set_xlim([2012, 2022])
    ax.set_xlabel("Years")
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title, fontsize=tit_s, fontweight='bold')
    ax.legend(loc="best", bbox_to_anchor=(1.01, 0.7), ncols=1, frameon=False, fontsize=lab_s)
    # fig.tight_layout(pad=0.5, h_pad=1.0)

# ===========================================================
def plot_validation(df_sys=None,
                    data=None,
                    metrices=None,
                    prec_avg=None,
                    crop_options=None,
                    background=None,
                    savefig=None,
                    combined=None):
    total_subplots = 2 + len(crop_options)
    if combined:
        figsize = (f_s['single'][0], f_s['single'][1] * total_subplots)
        fig, axes = plt.subplots(ncols=1, nrows=total_subplots, figsize=figsize)
    else:
        fig, axes = None, [plt.subplots(figsize=f_s['single'])[1] for _ in range(total_subplots)]
    x = list(df_sys.index)

    if background:
        for ax in axes:
            plot_wet_dry_bc(ax, prec_avg)

    # Saturated thickness
    ax = axes[0]
    v = 'GW_st'
    if df_sys is not None:
        ax.plot(x, df_sys[v], color=col[v], label="Sim")
    if data is not None:
        data_ = data.loc[x, :]
        ax.plot(x, data_[v], color=col["k"], ls="--", label="Obs")
    if metrices is not None:
        ax.text(0.5, 0.95, f"RMSE: {round(metrices.loc[v, 'rmse'], 3)}", transform=ax.transAxes,
                va='top', ha='center', fontsize=lab_s)
    plot_settings(fig,
                  ax,
                  y_label="(m)",
                  title="Saturated Thickness")

    ax.axhline(y=ax.get_ylim()[1], color=col["k"], linewidth=1)

    # Water use
    ax = axes[1]
    v = 'withdrawal'
    if df_sys is not None:
        ax.plot(x, df_sys[v], color=col[v], label="Sim")
    if data is not None:
        data_ = data.loc[x, :]
        ax.plot(x, (data_[v] / 100), color=col["k"], ls="--", label="Obs")
    if metrices is not None:
        ax.text(0.5, 0.95, f"RMSE: {round(metrices.loc[v, 'rmse'], 2)}", transform=ax.transAxes,
                va='top', ha='center', fontsize=lab_s)
    plot_settings(fig,
                  ax,
                  y_label="($1E^6$ $m^3$)",
                  title="Water Use")

    # Crop ratios
    c_hex_list = [col[crop] for crop in crop_options]
    for i, crop in enumerate(crop_options):
        ax = axes[2 + i]
        if df_sys is not None:
            ax.plot(x, df_sys[crop], color=c_hex_list[i], label="Sim")
        if data is not None:
            data_ = data.loc[x, :]
            ax.plot(x, data_[crop], color=col["k"], ls="--", label="Obs")
        if metrices is not None:
            ax.text(0.5, 0.95, f"RMSE: {round(metrices.loc[crop, 'rmse'], 3)}", transform=ax.transAxes,
                    va='top', ha='center', fontsize=lab_s)
        plot_settings(fig,
                      ax,
                      y_label="Ratio",
                      title=f"{crop.capitalize()} Yield")

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
# ===========================================================