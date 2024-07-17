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
cmap = plt.get_cmap("tab10")

def plot_cali_gwrc2(df_sys=None, data=None, metrices=None, prec_avg=None, stochastic=[], title=None, savefig=None):
    ##### st & withdrawal
    height_ratios = [1,1,1]
    fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(7, 5), sharex=True, sharey=False,
                             gridspec_kw={'height_ratios': height_ratios})
    axes = axes.flatten()
    x = list(df_sys.index)

    # Wet Dry background
    def plot_wet_dry_bc(ax):
        if prec_avg is not None:
            prec_avg["year"] = prec_avg.index
            mean_rainfall = prec_avg["annual"].mean()
            for index, row in prec_avg.iterrows():
                if row['annual'] <= mean_rainfall:
                    #ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color="peru", alpha=0.2, lw=0)
                    ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color="gray", alpha=0.2, lw=0)

    # Saturated thickness
    v = 'GW_st'
    c_hex = '#4472C4'
    ax = axes[0]
    if df_sys is not None:
        ax.plot(x, df_sys[v], c=c_hex, label="Sim")
    for df in stochastic:
        ax.plot(x, df[v], c=c_hex, alpha=0.3, lw=0.5, zorder=0)
    if data is not None:
        data_ = data.loc[x, :]
        ax.plot(x, data_[v], c="k", ls="--", label="Obv")
    if metrices is not None:
        ax.text(0.05, 0.05, f"RMSE: {round(metrices.loc[v, 'rmse'], 3)}", transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='left')
    plot_wet_dry_bc(ax)
    ax.legend(loc="upper right", ncols=2)
    ax.axvline(2012.5, c="red", ls=":", lw=1)
    ax.axvline(2017.5, c="grey", ls=":", lw=1)
    ax.set_xlim([2011, 2022])
    ax.set_ylabel("(a)\n\nSaturated\nthickness (m)", fontsize=12)

    # Withdrawal
    v = 'withdrawal'
    c_hex = '#2F5597'
    ax = axes[1]
    if df_sys is not None:
        ax.plot(x, df_sys[v], c=c_hex, label="Sim")
    for df in stochastic:
        ax.plot(x, df[v], c=c_hex, alpha=0.3, lw=0.5, zorder=0)
    if data is not None:
        data_ = data.loc[x, :]
        ax.plot(x, data_[v], c="k", ls="--", label="Obv")
    if metrices is not None:
        ax.text(0.05, 0.05, f"RMSE: {round(metrices.loc[v, 'rmse'], 3)}", transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='left')
    plot_wet_dry_bc(ax)
    ax.legend(loc="upper right", ncols=2)
    ax.axvline(2012.5, c="red", ls=":", lw=1)
    ax.axvline(2017.5, c="grey", ls=":", lw=1)
    ax.set_xlim([2011, 2022])
    ax.set_ylabel("(b)\n\nWithdrawal\n($10^6$ $m^3$)", fontsize=12)

    # State ratio
    ax = axes[2]
    states = ["Imitation", "Social comparison", "Repetition", "Deliberation"]
    c_hex_list = ['#ED7D31', '#996633', '#7030A0', '#FF85FF']
    if df_sys is not None:
        dff = df_sys[states]
        num_agt = sum(dff.iloc[0, :])
        dff = dff/num_agt
        for i, state in enumerate(states):
            ax.plot(x, dff[state], zorder=20, label=state, c=c_hex_list[i])

    if stochastic is not None:
        for df in stochastic:
            dff = df[states]
            num_agt = sum(dff.iloc[0, :])
            dff = dff/num_agt
            for i, state in enumerate(states):
                ax.plot(x, dff[state], zorder=1, c=c_hex_list[i], alpha=0.3, lw=0.5)
    plot_wet_dry_bc(ax)
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([0, 1.4])
    ax.set_ylabel("(c)\n\nCONSUMAT\nstate ratio", fontsize=12)
    ax.set_xlabel("")
    ax.legend(ncols=2, fontsize=9, labelspacing=0.3, loc="upper right")
    ax.axvline(2012.5, c="red", ls=":", lw=1)
    ax.axvline(2017.5, c="grey", ls=":")

    fig.align_ylabels(axes)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Year", fontsize=12)

    if title is not None:
        plt.title(title)

    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig, dpi=1000)
    plt.show()
#%%

def plot_crop_ratio(df_sys=None, data=None, metrices=None, prec_avg=None, title=None, stochastic=[],
                    crop_options=["corn", "others"], savefig=None):
    # Wet Dry background
    def plot_wet_dry_bc(ax):
        if prec_avg is not None:
            prec_avg["year"] = prec_avg.index
            mean_rainfall = prec_avg["annual"].mean()
            for index, row in prec_avg.iterrows():
                if row['annual'] <= mean_rainfall:
                    ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color="gray", alpha=0.2, lw=0)

    ##### Crop ratio
    c_hex_list = ['#FDC829', '#8D464C']
    abcde = ["a", "b"]
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(7, 5), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, crop in enumerate(crop_options):
        ax = axes[i]
        x = list(df_sys.index)
        if metrices is not None:
            rmse = round(metrices.loc[crop, 'rmse'], 3)
            if crop == "corn":
                ax.text(0.05, 0.05, f"RMSE: {rmse}", transform=ax.transAxes,
                        verticalalignment='bottom', horizontalalignment='left')
            else:
                ax.text(0.05, 0.80, f"RMSE: {rmse}", transform=ax.transAxes,
                        verticalalignment='bottom', horizontalalignment='left')
        if df_sys is not None:
            ax.plot(x, df_sys[crop], c=c_hex_list[i], zorder=100)
        if data is not None:
            data_ = data.loc[x, :]
            ax.plot(x, data_[crop], c="k", ls="--", zorder=1000)

        if stochastic is not None:
            for df in stochastic:
                ax.plot(x, df[crop], c=c_hex_list[i], zorder=1, alpha=0.2, lw=0.5)

        ax.set_title(f'({abcde[i]})\n'+crop.capitalize())
        ax.set_xlim([2011, 2022])
        ax.set_ylim([0, 1])
        ax.axvline(2012.5, c="red", ls=":", lw=1)
        ax.axvline(2017.5, c="grey", ls=":", lw=1)
        plot_wet_dry_bc(ax)
    fig.delaxes(axes[-1])
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Ratio\n", fontsize=12)
    line_obv = Line2D([0], [0], label='Obv', c='k', ls='--')
    line_sim = Line2D([0], [0], label='Sim', c='k', ls='-')
    plt.legend(handles=[line_obv, line_sim], loc="lower right", labelspacing=0.8, fontsize=8)

    if title is not None:
        plt.title(title+"\n")

    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig, dpi=1000)
    plt.show()