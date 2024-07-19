import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import pandas as pd
from copy import deepcopy
from adjustText import adjust_text

# Set the font properties globally
matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Helvetica')
matplotlib.rc('text', usetex='false')
matplotlib.rc('font', size=10)

# Colors
col = {"dry_year": "#C1C0C0",
       "GW_st": "#377EB8",
       "withdrawal": "#4472C4",
       "crop": "#4DAF4A",
       "corn": "#FDC829",
       "sorghum": "#8D464C",
       "soybeans": "#216D00",
       "wheat": "#A36D00",
       "fallow": "#3E3E3E",
       "others": "#4DAF4A",
       "profit": "#FFC000",
       "energy": "#E41A1C",
       "Imitation": "#13BDCF",
       "Social comparison": "#E27BC3",
       "Repetition": "#FF7F00",
       "Deliberation": "#984EA3",
       "k": "#000000"
}

f_s = {
    "single": (5,2.5),
    "multiple": (5,5)
}
def plot_GW_st(df_sys=None, data=None, metrices=None, prec_avg=None, crop_options=None, savefig=None):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=f_s['single'])
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    x = list(df_sys.index)

    # Wet Dry background
    def plot_wet_dry_bc(ax):
        if prec_avg is not None:
            prec_avg["year"] = prec_avg.index
            mean_rainfall = prec_avg["annual"].mean()
            for index, row in prec_avg.iterrows():
                if row['annual'] <= mean_rainfall:
                    ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color=col["dry_year"], alpha=0.5, lw=0)

    # Saturated thickness
    v = 'GW_st'
    plot_wet_dry_bc(ax)
    if df_sys is not None:
        ax.plot(x, df_sys[v], color=col[v], label="Sim", zorder=1000)
    if data is not None:
        data_ = data.loc[x, :]
        ax.plot(x, data_[v], color=col["k"], ls="--", label="Obv", zorder=100)
    if metrices is not None:
        ax.text(0.95, 0.1, f"RMSE: {round(metrices.loc[v, 'rmse'], 3)}", transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right', fontsize=8)
    ax.legend(loc="upper right", ncols=2, frameon=False)
    ax.axvline(2013, color=col["k"], ls=":", lw=1)
    ax.axvline(2018, color=col["k"], ls=":", lw=1)
    ax.set_xlim([2012, 2022])
    ax.set_xlabel("Years")
    ax.set_ylabel("(m)")
    ax.set_title("Saturated thickness")

    if savefig is not None:
        plt.savefig(savefig, dpi=1000)

    plt.show()

def plot_withdrawal(df_sys=None, data=None, metrices=None, prec_avg=None, crop_options=None, savefig=None):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=f_s['single'])
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    x = list(df_sys.index)

    # Wet Dry background
    def plot_wet_dry_bc(ax):
        if prec_avg is not None:
            prec_avg["year"] = prec_avg.index
            mean_rainfall = prec_avg["annual"].mean()
            for index, row in prec_avg.iterrows():
                if row['annual'] <= mean_rainfall:
                    ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color=col["dry_year"], alpha=0.5, lw=0)

    # Withdrawal
    v = 'withdrawal'
    plot_wet_dry_bc(ax)
    if df_sys is not None:
        ax.plot(x, df_sys[v], color=col[v], label="Sim", zorder=1000)
    if data is not None:
        data_ = data.loc[x, :]
        ax.plot(x, data_[v], color=col["k"], ls="--", label="Obv", zorder=100)
    if metrices is not None:
        ax.text(0.95, 0.1, f"RMSE: {round(metrices.loc[v, 'rmse'], 3)}", transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right', fontsize=8)
    ax.legend(loc="upper right", ncols=2, frameon=False)
    ax.axvline(2013, color=col["k"], ls=":", lw=1)
    ax.axvline(2018, color=col["k"], ls=":", lw=1)
    ax.set_xlim([2012, 2022])
    ax.set_xlabel("Years")
    ax.set_ylabel("($10^6$ $m^3$)")
    ax.set_title("Water use")

    if savefig is not None:
        plt.savefig(savefig, dpi=1000)

    plt.show()

def plot_crop_ratio(df_sys=None, data=None, metrices=None, prec_avg=None, crop_options=None, savefig=None):
    fig, axes = plt.subplots(ncols=1, nrows=len(crop_options), figsize=f_s['multiple'])
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if len(crop_options) == 1:
        axes = [axes]
    x = list(df_sys.index)

    # Wet Dry background
    def plot_wet_dry_bc(ax):
        if prec_avg is not None:
            prec_avg["year"] = prec_avg.index
            mean_rainfall = prec_avg["annual"].mean()
            for index, row in prec_avg.iterrows():
                if row['annual'] <= mean_rainfall:
                    ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color=col["dry_year"], alpha=0.5, lw=0)

    # Crop ratios
    c_hex_list = [col[crop] for crop in crop_options]

    for i, crop in enumerate(crop_options):
        x = list(df_sys.index)
        ax = axes[i]
        if df_sys is not None:
            ax.plot(x, df_sys[crop], color=c_hex_list[i], zorder=1000, label="Sim")
        if data is not None:
            data_ = data.loc[x, :]
            ax.plot(x, data_[crop], color=col["k"], ls="--", zorder=100, label= "Obs")
        if metrices is not None:
            ax.text(0.95, 0.1, f"RMSE: {round(metrices.loc[crop, 'rmse'], 3)}", transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right', fontsize=8)
        ax.set_title(crop.capitalize())
        ax.set_xlim([2012, 2022])
        ax.set_ylim([0, 1])
        ax.axvline(2013, color=col["k"], ls=":", lw=1)
        ax.axvline(2018, color=col["k"], ls=":", lw=1)
        plot_wet_dry_bc(ax)
    if len(crop_options) > 1:
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Years")
        plt.ylabel("Ratio")
        line_obv = Line2D([0], [0], label='Obv', color=col["k"], ls='--')
        line_sim = Line2D([0], [0], label='Sim', color=col["k"], ls='-')
        plt.legend(handles=[line_obv, line_sim], loc="upper right", ncols=len(crop_options), frameon=False)

    if savefig is not None:
        plt.savefig(savefig, dpi=1000)

    fig.tight_layout(pad=0.7)
    plt.show()

def plot_rainfed(df_sys=None, data=None, metrices=None, prec_avg=None, crop_options=None, savefig=None):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=f_s['single'])
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    x = list(df_sys.index)

    # Wet Dry background
    def plot_wet_dry_bc(ax):
        if prec_avg is not None:
            prec_avg["year"] = prec_avg.index
            mean_rainfall = prec_avg["annual"].mean()
            for index, row in prec_avg.iterrows():
                if row['annual'] <= mean_rainfall:
                    ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color=col["dry_year"], alpha=0.5, lw=0)

    # Rainfed fields
    v = 'rainfed'
    plot_wet_dry_bc(ax)
    if df_sys is not None:
        ax.plot(x, df_sys[v], color=col["crop"], label="Sim", zorder=1000)
    if data is not None:
        data_ = data.loc[x, :]
        ax.plot(x, data_[v], color=col["k"], ls="--", label="Obv", zorder=100)
    if metrices is not None:
        ax.text(0.95, 0.1, f"RMSE: {round(metrices.loc[v, 'rmse'], 3)}", transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right', fontsize=8)
    ax.legend(loc="upper right", ncols=2, frameon=False)
    ax.axvline(2013, color=col["k"], ls=":", lw=1)
    ax.axvline(2018, color=col["k"], ls=":", lw=1)
    ax.set_xlim([2012, 2022])
    ax.set_xlabel("Years")
    ax.set_ylabel("Ratio")
    ax.set_title("Rainfed fields")

    if savefig is not None:
        plt.savefig(savefig, dpi=1000)

    plt.show()

def plot_aquifer(df_sys=None, data=None, metrices=None, prec_avg=None, crop_options=None, savefig=None):
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=f_s['multiple'])
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    x = list(df_sys.index)

    # Wet Dry background
    def plot_wet_dry_bc(ax):
        if prec_avg is not None:
            prec_avg["year"] = prec_avg.index
            mean_rainfall = prec_avg["annual"].mean()
            for index, row in prec_avg.iterrows():
                if row['annual'] <= mean_rainfall:
                    ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color=col["dry_year"], alpha=0.5, lw=0)

    # Aquifer - irrigation depth total and by crops
    ax1 = axes[0]
    plot_wet_dry_bc(ax1)
    c_hex_list = [col[crop] for crop in crop_options]
    crop_irr_depth_cols = [f"{crop}_irr_depth" for crop in crop_options]
    if df_sys is not None:
        ax1.stackplot(x, df_sys[crop_irr_depth_cols].T, labels=crop_options, colors=c_hex_list)

    proxy_artists = [Patch(facecolor=c_hex_list[i]) for i in range(len(c_hex_list))] + [Line2D([0], [0], color=col["withdrawal"], lw=2)]

    v = 'total_irr_depth'
    if df_sys is not None:
        ax1.plot(x, df_sys[v], color=col["withdrawal"], label="Total", lw=2)
    ax1.axvline(2013, color=col["k"], ls=":", lw=1)
    ax1.axvline(2018, color=col["k"], ls=":", lw=1)
    ax1.legend(proxy_artists, crop_options + ["Total"], loc="upper right", ncols=len(crop_options)+1, fontsize = 8)
    ax1.set_xlim([2012, 2022])
    ax1.set_xlabel("Years")
    ax1.set_ylabel("(cm)")
    ax1.set_title("Irrigation depth")

    # Aquifer - irrigation volumen total and by crops
    ax2 = axes[1]
    plot_wet_dry_bc(ax2)
    c_hex_list = [col[crop] for crop in crop_options]
    crop_irr_vol_cols = [f"{crop}_irr_vol" for crop in crop_options]
    if df_sys is not None:
        ax2.stackplot(x, df_sys[crop_irr_vol_cols].T, labels=crop_options, colors=c_hex_list)

    proxy_artists = [Patch(facecolor=c_hex_list[i]) for i in range(len(c_hex_list))] + [Line2D([0], [0], color=col["withdrawal"], lw=2)]

    v = 'total_irr_vol'
    if df_sys is not None:
        ax2.plot(x, df_sys[v], color=col["withdrawal"], label="Total", lw=2)
    ax2.axvline(2013, color=col["k"], ls=":", lw=1)
    ax2.axvline(2018, color=col["k"], ls=":", lw=1)
    ax2.legend(proxy_artists, crop_options + ["Total"], loc="upper right", ncols=len(crop_options)+1, fontsize = 8)
    ax2.set_xlim([2012, 2022])
    ax2.set_xlabel("Years")
    ax2.set_ylabel("(m-ha)")
    ax2.set_title("Irrigation volume")

    fig.suptitle("AQUIFER", color=col['withdrawal'], fontweight='bold', horizontalalignment='center', va='top')

    if savefig is not None:
        plt.savefig(savefig, dpi=1000)

    fig.tight_layout(pad=0.7)
    plt.show()

def plot_field(df_sys=None, data=None, metrices=None, prec_avg=None, crop_options=None, savefig=None):
    fig, axes = plt.subplots(ncols=1, nrows=len(crop_options), figsize=f_s['multiple'])
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if len(crop_options) == 1:
        axes = [axes]
    x = list(df_sys.index)

    # Wet Dry background
    def plot_wet_dry_bc(ax):
        if prec_avg is not None:
            prec_avg["year"] = prec_avg.index
            mean_rainfall = prec_avg["annual"].mean()
            for index, row in prec_avg.iterrows():
                if row['annual'] <= mean_rainfall:
                    ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color=col["dry_year"], alpha=0.5, lw=0)

    # Field - yield per crop
    c_hex_list = [col[crop] for crop in crop_options]
    for i, crop in enumerate(crop_options):
        x = list(df_sys.index)
        ax = axes[i]
        yield_col = f"{crop}_yield"
        if df_sys is not None:
            ax.plot(x, df_sys[yield_col], color=c_hex_list[i], label=crop)
        ax.set_title(crop.capitalize())
        ax.set_xlim([2012, 2022])
        ax.axvline(2013, color=col["k"], ls=":", lw=1)
        ax.axvline(2018, color=col["k"], ls=":", lw=1)
        plot_wet_dry_bc(ax)
    if len(crop_options) > 1:
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Years")
        plt.ylabel("(1e4 bu)")
        fig.suptitle("FIELD", color=col['crop'], fontweight='bold', horizontalalignment='center', va='top')

    if savefig is not None:
        plt.savefig(savefig, dpi=1000)

    fig.tight_layout(pad=0.7)
    plt.show()

def plot_well(df_sys=None, data=None, metrices=None, prec_avg=None, crop_options=None, savefig=None):
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=f_s['multiple'])
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    x = list(df_sys.index)

    # Wet Dry background
    def plot_wet_dry_bc(ax):
        if prec_avg is not None:
            prec_avg["year"] = prec_avg.index
            mean_rainfall = prec_avg["annual"].mean()
            for index, row in prec_avg.iterrows():
                if row['annual'] <= mean_rainfall:
                    ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color=col["dry_year"], alpha=0.5, lw=0)

    # Well - total energy
    ax1 = axes[0]
    v = 'total_energy'
    plot_wet_dry_bc(ax1)
    if df_sys is not None:
        ax1.plot(x, df_sys[v], color=col["energy"])
    # ax1.legend(loc="upper right", ncols=2)
    ax1.axvline(2013, color=col["k"], ls=":", lw=1)
    ax1.axvline(2018, color=col["k"], ls=":", lw=1)
    ax1.set_xlim([2012, 2022])
    ax1.set_xlabel("Years")
    ax1.set_ylabel("(PJ)")
    ax1.set_title("Energy")

    # Well - energy per irrigation depth
    ax2 = axes[1]
    v = 'energy_irr_depth'
    plot_wet_dry_bc(ax2)
    if df_sys is not None:
        ax2.plot(x, df_sys[v], color=col["energy"])
    # ax2.legend(loc="upper right", ncols=2)
    ax2.axvline(2013, color=col["k"], ls=":", lw=1)
    ax2.axvline(2018, color=col["k"], ls=":", lw=1)
    ax2.set_xlim([2012, 2022])
    ax2.set_xlabel("Years")
    ax2.set_ylabel("(PJ/cm)")
    ax2.set_title("Energy per irrigation depth")

    # Well - energy per irrigation volumen
    ax3 = axes[1]
    v = 'energy_irr_vol'
    plot_wet_dry_bc(ax2)
    if df_sys is not None:
        ax2.plot(x, df_sys[v], color=col["energy"])
    # ax2.legend(loc="upper right", ncols=2)
    ax3.axvline(2013, color=col["k"], ls=":", lw=1)
    ax3.axvline(2018, color=col["k"], ls=":", lw=1)
    ax3.set_xlim([2012, 2022])
    ax3.set_xlabel("Years")
    ax3.set_ylabel("(PJ/m-ha)")
    ax3.set_title("Energy per irrigation depthvol")

    fig.suptitle("WELL", color=col['energy'], fontweight='bold', horizontalalignment='center', va='top')

    if savefig is not None:
        plt.savefig(savefig, dpi=1000)

    fig.tight_layout(pad=0.7)
    plt.show()

def plot_finance(df_sys=None, data=None, metrices=None, prec_avg=None, crop_options=None, savefig=None):
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=f_s['multiple'])
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    x = list(df_sys.index)

    # Wet Dry background
    def plot_wet_dry_bc(ax):
        if prec_avg is not None:
            prec_avg["year"] = prec_avg.index
            mean_rainfall = prec_avg["annual"].mean()
            for index, row in prec_avg.iterrows():
                if row['annual'] <= mean_rainfall:
                    ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color=col["dry_year"], alpha=0.5, lw=0)

    # Well - total energy
    ax1 = axes[0]
    v = 'total_profit'
    plot_wet_dry_bc(ax1)
    if df_sys is not None:
        ax1.plot(x, df_sys[v], color=col["energy"])
    # ax1.legend(loc="upper right", ncols=2)
    ax1.axvline(2013, color=col["k"], ls=":", lw=1)
    ax1.axvline(2018, color=col["k"], ls=":", lw=1)
    ax1.set_xlim([2012, 2022])
    ax1.set_xlabel("Years")
    ax1.set_ylabel("(1e4 $)")
    ax1.set_title("Energy")

    # Well - energy per irrigation depth
    ax2 = axes[1]
    v = 'energy_irr_depth'
    plot_wet_dry_bc(ax2)
    if df_sys is not None:
        ax2.plot(x, df_sys[v], color=col["energy"])
    # ax2.legend(loc="upper right", ncols=2)
    ax2.axvline(2013, color=col["k"], ls=":", lw=1)
    ax2.axvline(2018, color=col["k"], ls=":", lw=1)
    ax2.set_xlim([2012, 2022])
    ax2.set_xlabel("Years")
    ax2.set_ylabel("(1e4 $/cm)")
    ax2.set_title("Profit per irrigation depth")
    fig.suptitle("FINANCE", color=col['profit'], fontweight='bold', horizontalalignment='center', va='top')

    if savefig is not None:
        plt.savefig(savefig, dpi=1000)

    fig.tight_layout(pad=0.7)
    plt.show()

def plot_behavior(df_sys=None, data=None, metrices=None, prec_avg=None, crop_options=None, savefig=None):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=f_s['single'])
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    x = list(df_sys.index)

    # Wet Dry background
    def plot_wet_dry_bc(ax):
        if prec_avg is not None:
            prec_avg["year"] = prec_avg.index
            mean_rainfall = prec_avg["annual"].mean()
            for index, row in prec_avg.iterrows():
                if row['annual'] <= mean_rainfall:
                    ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color=col["dry_year"], alpha=0.5, lw=0)

    # State ratio
    states = ["Imitation", "Social comparison", "Repetition", "Deliberation"]
    plot_wet_dry_bc(ax)
    c_hex_list = [col[state] for state in states]
    if df_sys is not None:
        dff = df_sys[states]
        num_agt = sum(dff.iloc[0, :])
        dff = dff / num_agt
        for i, state in enumerate(states):
            ax.plot(x, dff[state], zorder=20, color=c_hex_list[i], label=state)
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([0, 1.4])
    ax.set_ylabel("Farmers")
    ax.set_xlabel("Years")
    ax.legend(loc="upper right", ncols=len(states), frameon=False)
    ax.axvline(2013, color=col["k"], ls=":", lw=1)
    ax.axvline(2018, color=col["k"], ls=":")
    ax.set_title("CONSUMAT States")
    fig.suptitle("BEHAVIOR", color=col['Repetition'], fontweight='bold', horizontalalignment='center', va='top')

    if savefig is not None:
        plt.savefig(savefig, dpi=1000)
    plt.show()