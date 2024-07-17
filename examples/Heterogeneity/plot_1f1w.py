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
#plt.rcParams['font.family'] = 'Helvetica Neue' # e.g., 'Arial', 'Helvetica', 'sans-serif'
cmap = plt.get_cmap("tab10")

def plot_cali_gwrc(df_sys=None, data=None, metrices=None, prec_avg=None, stochastic=[], title=None, savefig=None):
    ##### st & withdrawal
    fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(5.5, 5), sharex=True, sharey=False)
    axes = axes.flatten()
    x = list(df_sys.index)

    for i, v in enumerate(['GW_st', 'withdrawal']):
        ax = axes[i]
        if v == 'withdrawal':  # mha to 10^6 m3
            if df_sys is not None:
                ax.plot(x, df_sys[v]/100, c=cmap(i+5), label="Sim")
            for df in stochastic:
                ax.plot(x, df[v]/100, c=cmap(i+5), alpha=0.3, lw=0.5, zorder=0)
            if data is not None:
                data_ = data.loc[x, :]
                ax.plot(x, data_[v]/100, c="k", ls="--", label="Obv")
        else:
            if df_sys is not None:
                ax.plot(x, df_sys[v], c=cmap(i+5), label="Sim")
            for df in stochastic:
                ax.plot(x, df[v], c=cmap(i+5), alpha=0.3, lw=0.5, zorder=0)
            if data is not None:
                data_ = data.loc[x, :]
                ax.plot(x, data_[v], c="k", ls="--", label="Obv")
        if metrices is not None:
            ax.text(0.05, 0.05, f"RMSE: {round(metrices.loc[v, 'rmse'], 3)}", transform=ax.transAxes,
                    verticalalignment='bottom', horizontalalignment='left')

        # Add the background color for the dry year
        if prec_avg is not None:
            prec_avg["year"] = prec_avg.index
            mean_rainfall = prec_avg["annual"].mean()
            for index, row in prec_avg.iterrows():
                if row['annual'] <= mean_rainfall:
                    ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color="peru", alpha=0.2, lw=0)
                # if row['annual'] > mean_rainfall:
                #     ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color="deepskyblue", alpha=0.2, lw=0)
        ax.legend(loc="upper right", ncols=2)
        ax.axvline(2012.5, c="red", ls=":", lw=1)
        ax.axvline(2017.5, c="grey", ls=":", lw=1)
        ax.set_xlim([2011, 2022])
        ylabels = ["Saturated\nthickness (m)", "Withdrawal\n($10^6$ $m^3$)"]
        ax.set_ylabel(ylabels[i], fontsize=12)

    # state ratio
    ax = axes[2]
    states = ["Imitation", "Social comparison", "Repetition", "Deliberation"]

    if df_sys is not None:
        dff = df_sys[states]
        num_agt = sum(dff.iloc[0, :])
        dff = dff/num_agt
        for i, state in enumerate(states):
            ax.plot(x, dff[state], zorder=20, label=state, c=cmap(i))

    if stochastic is not None:
        for df in stochastic:
            dff = df[states]
            num_agt = sum(dff.iloc[0, :])
            dff = dff/num_agt
            for i, state in enumerate(states):
                ax.plot(x, dff[state], zorder=1, c=cmap(i), alpha=0.3, lw=0.5)

    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([0, 1.4])
    ax.set_ylabel("CONSUMAT\nstate ratio", fontsize=12)
    ax.set_xlabel("")
    ax.legend(ncols=2, fontsize=9, labelspacing=0.3, loc="upper right")
    ax.axvline(2012.5, c="red", ls=":", lw=1)
    ax.axvline(2017.5, c="grey", ls=":")
    # Add the background color for the wet year
    if prec_avg is not None:
        prec_avg["year"] = prec_avg.index
        mean_rainfall = prec_avg["annual"].mean()
        for index, row in prec_avg.iterrows():
            if row['annual'] <= mean_rainfall:
                ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color="peru", alpha=0.2, lw=0)
            # if row['annual'] > mean_rainfall:
            #     ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color="deepskyblue", alpha=0.2, lw=0)

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

# dfs_sys = []
# for m in m_list:
#     df_farmers_lema, df_fields_lema, df_wells_lema, df_aquifers_lema = SD6Model.get_dfs(m)
#     df_sys_lema = SD6Model.get_df_sys(m, df_farmers_lema, df_fields_lema, df_wells_lema, df_aquifers_lema)
#     dfs_sys.append(df_sys_lema)
# plot_cali_gwrc(None, data, None, prec_avg, stochastic=dfs_sys, savefig=None)
#%%
def plot_cali_gwrc2(df_sys=None, data=None, metrices=None, prec_avg=None, stochastic=[], title=None, savefig=None):
    ##### st & withdrawal
    height_ratios = [1,1,1.4,0.8,0.8]
    fig, axes = plt.subplots(ncols=1, nrows=5, figsize=(5.5, 6.5), sharex=True, sharey=False,
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

    # Profit
    v = 'Profit per water use'
    c_hex = '#FFC000'
    ax = axes[3]
    if df_sys is not None:
        ax.plot(x, df_sys[v], c=c_hex, label="Sim")
    for df in stochastic:
        ax.plot(x, df[v], c=c_hex, alpha=0.3, lw=0.5, zorder=0)
    plot_wet_dry_bc(ax)
    ax.legend(loc="upper left", ncols=2)
    ax.axvline(2012.5, c="red", ls=":", lw=1)
    ax.axvline(2017.5, c="grey", ls=":", lw=1)
    ax.set_xlim([2011, 2022])
    ax.set_ylabel("(d)\nProfit per\nwater use\n($10^4$/cm)", fontsize=12)

    # Energy
    v = 'Energy per water use'
    c_hex = '#C00000'
    ax = axes[4]
    if df_sys is not None:
        ax.plot(x, df_sys[v], c=c_hex, label="Sim")
    for df in stochastic:
        ax.plot(x, df[v], c=c_hex, alpha=0.3, lw=0.5, zorder=0)
    plot_wet_dry_bc(ax)
    ax.legend(loc="upper left", ncols=2)
    ax.axvline(2012.5, c="red", ls=":", lw=1)
    ax.axvline(2017.5, c="grey", ls=":", lw=1)
    ax.set_xlim([2011, 2022])
    ax.set_ylabel("(e)\nProfit per\nwater use\n($10^4$/cm)", fontsize=12)

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
                    #ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color="peru", alpha=0.2, lw=0)
                    ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color="gray", alpha=0.2, lw=0)

    ##### Crop ratio
    c_hex_list = ['#FDC829', '#8D464C']
    abcde = ["a", "b"]
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(5, 4), sharex=True, sharey=True)
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
    #blue_rect = Patch(facecolor=cmap(0), edgecolor=cmap(0), alpha= 0.3, label='Available\nprecipitation')
    #line_sim_no = Line2D([0], [0], label='Sim\n(no LEMA)', c='k', ls=':')
    plt.legend(handles=[line_obv, line_sim], loc="lower right", labelspacing=0.8, fontsize=8)

    if title is not None:
        plt.title(title+"\n")

    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig, dpi=1000)
    plt.show()

#plot_crop_ratio(None, data, None, prec_avg, stochastic=dfs_sys, savefig=None)
#%%
def reg_prec_withdrawal(prec_avg, df_sys, df_sys_nolema=None, data=None,
                        df_sys_list=None, df_sys_nolema_list=None,
                        dot_labels=False, obv_dots=False, savefig=None):
    # Split the dataframe based on years
    prec_2011_2012 = prec_avg.loc[2011:2012, 'annual']
    step = (max(prec_2011_2012)-min(prec_2011_2012))/100
    x_2011_2012 = np.arange(min(prec_2011_2012), max(prec_2011_2012)+step, step)
    prec_2013_2022 = prec_avg.loc[2013:2022, 'annual']
    step = (max(prec_2013_2022)-min(prec_2013_2022))/100
    x_2013_2022 = np.arange(min(prec_2013_2022), max(prec_2013_2022)+step, step)


    def add_labels(txt, x, y, color='blue', label_list=[]):
        txt=list(txt)
        x=list(x)
        y=list(y)
        for i in range(len(txt)):
            label_list.append(ax.text(x[i]+1, y[i]+1, txt[i], color=color, fontsize=5, alpha=0.5))
            #ax.annotate(txt[i], (x[i]+1, y[i]+1), color=color, fontsize=6)
        return label_list

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(5.5, 4))
    label_list=[]
    df_sys_withdrawal_2011_2012 = df_sys.loc[2011:2012, 'withdrawal']/100
    m1, b1 = np.polyfit(prec_2011_2012, df_sys_withdrawal_2011_2012, 1)
    ax.scatter(prec_2011_2012, df_sys_withdrawal_2011_2012, color='k', label='Pre-LEMA (2011-2012)', alpha=0.6)
    if dot_labels:
        label_list = add_labels(df_sys_withdrawal_2011_2012.index, x=prec_2011_2012, y=df_sys_withdrawal_2011_2012, color='k', label_list=label_list)
    ax.plot(x_2011_2012, m1*x_2011_2012 + b1, color='k')
    if df_sys_list is not None:
        lines = []
        for df in df_sys_list:
            m1, b1 = np.polyfit(prec_2011_2012, df.loc[2011:2012, 'withdrawal']/100, 1)
            lines.append(m1*x_2011_2012 + b1)
        lines = np.array(lines)
        lines_mean = np.mean(lines, axis=0)
        lines_std = np.std(lines, axis=0)
        ci = 2 * lines_std
        ax.fill_between(x_2011_2012, lines_mean - ci, lines_mean + ci, color='k', alpha=0.15)

    df_sys_withdrawal_2013_2022 = df_sys.loc[2013:2022, 'withdrawal']/100
    m1, b1 = np.polyfit(prec_2013_2022, df_sys_withdrawal_2013_2022, 1)
    ax.scatter(prec_2013_2022, df_sys_withdrawal_2013_2022, color='blue', label='With LEMA (2013-2022)', alpha=0.6)
    if dot_labels:
        label_list = add_labels(df_sys_withdrawal_2013_2022.index, x=prec_2013_2022, y=df_sys_withdrawal_2013_2022, color='blue', label_list=label_list)
    ax.plot(x_2013_2022, m1*x_2013_2022 + b1, color='blue')
    if df_sys_list is not None:
        lines = []
        for df in df_sys_list:
            m1, b1 = np.polyfit(prec_2013_2022, df.loc[2013:2022, 'withdrawal']/100, 1)
            lines.append(m1*x_2013_2022 + b1)
        lines = np.array(lines)
        lines_mean = np.mean(lines, axis=0)
        lines_std = np.std(lines, axis=0)
        ci = 2 * lines_std
        ax.fill_between(x_2013_2022, lines_mean - ci, lines_mean + ci, color='blue', alpha=0.15)


    if df_sys_nolema is not None:
        df_sys_withdrawal_2013_2022 = df_sys_nolema.loc[2013:2022, 'withdrawal']/100
        m1, b1 = np.polyfit(prec_2013_2022, df_sys_withdrawal_2013_2022, 1)
        ax.scatter(prec_2013_2022, df_sys_withdrawal_2013_2022, color='red', label='No LEMA (2013-2022)', alpha=0.6)
        if dot_labels:
            label_list = add_labels(df_sys_withdrawal_2013_2022.index, x=prec_2013_2022, y=df_sys_withdrawal_2013_2022, color='red', label_list=label_list)
        ax.plot(x_2013_2022, m1*x_2013_2022 + b1, color='red')
        if df_sys_list is not None:
            lines = []
            for df in df_sys_nolema_list:
                m1, b1 = np.polyfit(prec_2013_2022, df.loc[2013:2022, 'withdrawal']/100, 1)
                lines.append(m1*x_2013_2022 + b1)
            lines = np.array(lines)
            lines_mean = np.mean(lines, axis=0)
            lines_std = np.std(lines, axis=0)
            ci = 2 * lines_std
            ax.fill_between(x_2013_2022, lines_mean - ci, lines_mean + ci, color='red', alpha=0.15)

    if data is not None:
        m1, b1 = np.polyfit(prec_2011_2012, data.loc[2011:2012, 'withdrawal']/100, 1)
        ax.plot(x_2011_2012, m1*x_2011_2012 + b1, color='k', ls="--", alpha=0.5, lw=0.5)

        m2, b2 = np.polyfit(prec_2013_2022, data.loc[2013:2022, 'withdrawal']/100, 1)
        ax.plot(x_2013_2022, m2*x_2013_2022 + b2, color='blue', ls="--", alpha=0.5, lw=0.5)

        if obv_dots:
            ax.scatter(prec_2011_2012, data.loc[2011:2012, 'withdrawal']/100, color='k',
                       label='Observation', alpha=0.6, marker="+")
            ax.scatter(prec_2013_2022, data.loc[2013:2022, 'withdrawal']/100,
                       color='blue', alpha=0.4, marker="+")  # , label='Post-LEMA (obv)'


    # Legend, labels, title, and show
    adjust_text(label_list, ax=ax, force_static=(0.5, 3))
    ax.legend()
    ax.set_xlabel('Annual precipitation (cm)')
    ax.set_ylabel('Annual withdrawal ($10^6$ $m^3$)')
    if savefig is not None:
        plt.savefig(savefig, dpi=1000)
    plt.show()

# reg_prec_withdrawal(prec_avg, df_sys, df_sys_nolema=df_sys, data=data,
#                     df_sys_list=dfs_sys, df_sys_nolema_list=dfs_sys, savefig=None)
