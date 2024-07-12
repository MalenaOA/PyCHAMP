import os
import sys
import dill
import pandas as pd
from joblib.externals.loky import set_loky_pickler
from py_champ.models.sd6_model_1f1w_ci import SD6Model_1f1w_ci
join = os.path.join
#pc = "CL" #"ResearchPC"
wd = r"C:/Users/cecil/Dropbox/phD_Cecilia/Projects/Crop Insurance ERL/Codes/experiments/Data"
sys.setrecursionlimit(10000)
set_loky_pickler("dill")    # joblib
#os.chdir(rf"C:/Users/cecil/Dropbox/phD_Cecilia/Projects/Crop Insurance ERL/Codes/experiments")

# Load inputs and data
with open(join(wd, "Inputs", "Inputs_SD6_2012_2022_ci_test2011.pkl"), "rb") as f:
    (aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict,
     prec_aw_step, crop_price_step, harvest_price_step, projected_price_step,
     aph_revenue_based_coef_step) = dill.load(f)


#%%
data = pd.read_csv(join(wd, "Inputs", "Data_SD6_2012_2022_ci_test2011.csv"), index_col=["year"])
prec_avg = pd.read_csv(join(wd, "Inputs", "prec_avg_2011_2022.csv"), index_col=[0]).iloc[1:, :]
# General model settings
crop_options = ["corn", "sorghum", "soybeans", "wheat", "fallow"]

pars = {'perceived_risk': 0.54079728, 
        'forecast_trust': 0.5799478, 
        'sa_thre': 0.17005009, 
        'un_thre': 0.23259658} 
# pars = {'perceived_risk': 0.7539013390415119,
#         'forecast_trust': 0.8032197483934305,
#         'sa_thre': 0.14215821111637678,
#         'un_thre': 0.0773514357873846}

#%%
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
    seed=56,
    activate_ci=True,
    # kwargs
    gurobi_dict={"LogToConsole": 0, "NonConvex": 2, "Presolve": -1},
    crop_price_step=crop_price_step,
    harvest_price_step=harvest_price_step,
    projected_price_step=projected_price_step,
    aph_revenue_based_coef_step=aph_revenue_based_coef_step
    )
#%%
#m.step()
#%%
for i in range(11):
    m.step()

m.end()

df_sys, df_agt, df_other = m.get_dfs(m)

df_sys.plot()

df_sys_nor = (df_sys-df_sys.min())/(df_sys.max()-df_sys.min())
df_sys_nor.plot()

#%% General figure setting
# Set global font properties
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
fontsize = 16
plt.rcParams['font.size'] = fontsize  # sets the default font size
plt.rcParams['font.family'] = 'Arial'  # sets the default font family

# Set figure width
figwd = {"1": 140/1.5/25.4, "1.5": 140/25.4, "2": 140/1.5/25.4*2}

# Colors
colors = {"dry_year": "gainsboro",
          "GW_st": "#4472C4",
          "withdrawal": "#4472C4",
          "Well": "#C00000",
          "Profit": "#FFC000",
          "Imitation": "#17BECF",
          "Social comparison": "#E377C2",
          "Repetition": "#FF7F0E",
          "Deliberation": "#9467BD",
          "corn": "#F6BB00",
          "sorghum": "#8D464C",
          "soybeans": "#267000",
          "wheat": "#A57000",
          "fallow": "#404040",
          "others": "#404040",
          }

# General functions
def plot_wet_dry_bg(ax):
    prec_avg["year"] = prec_avg.index
    mean_rainfall = prec_avg["annual"].mean()
    for index, row in prec_avg.iterrows():
        if row['annual'] <= mean_rainfall:
            ax.axvspan(row['year'] - 0.5, row['year'] + 0.5, color=colors["dry_year"], lw=0, zorder=0)
    return ax

#%% Fig: Time series for saturated thickness, withdrawal, profit, energy, and CONSUMAT ratio
fontsize = 16
plt.rcParams['font.size'] = fontsize
def plot_main_ts(df_sys, data=None, metrices=None, wet_dry_bg=True, stochastic=[], savefig=None):
    fig = plt.figure(figsize=(figwd["1.5"], figwd["1.5"]*1.3))
    axes = []
    x = list(df_sys.index)

    def add_vlines(ax):
        ax.axvline(2013, c="red", ls=":", lw=1, zorder=1)
        ax.axvline(2018, c="grey", ls=":", lw=1, zorder=1)
        #ax.axvline(2019, c="k", ls="-.", lw=1.1, zorder=1)
        ax.set_xlim([2012, 2022])

    lw_sim = 2.5; lw_obv = 1; lw_sto = 0.7
    ylabel_xloc = -0.12
    # Saturated thickness
    v = 'GW_st'
    ax = fig.add_axes([0, 1/4*4+1/4/3*2.5, 1, 1/4])#axes[0]
    axes.append(ax)
    ax.plot([0]*len(df_sys[v]), df_sys[v], label="Sim", lw=lw_sim, c="k") # psuedo line
    ax.plot(x, df_sys[v], c=colors[v], lw=lw_sim, zorder=10)
    for df in stochastic:
        ax.plot(x, df[v], c=colors[v], alpha=0.3, lw=lw_sto, zorder=5)
    if data is not None:
        data_ = data.loc[x, :]
        ax.plot(x, data_[v], c="k", ls="--", label="Obv", lw=lw_obv, zorder=20)
    if metrices is not None:
        ax.text(0.05, 0.05, f"RMSE: {round(metrices.loc[v, 'rmse'], 3)}", transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='left')
    if wet_dry_bg:
        plot_wet_dry_bg(ax)

    handles, labels = ax.get_legend_handles_labels()
    rect = Patch(facecolor=colors["dry_year"], edgecolor=None, label='Dry year')
    handles.append(rect)
    labels.append('Dry year')
    ax.legend(handles, labels, loc="upper right", ncols=3, bbox_to_anchor=(1, 1.4), frameon=False)

    add_vlines(ax)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.set_ylabel("(a)\nSaturated\nthickness\n(m)")
    ax.yaxis.set_label_coords(ylabel_xloc, 0.5)
    
    # Set the y-axis limits
    ax.set_ylim(17, 20)

    # Withdrawal
    v = 'withdrawal'
    ax = fig.add_axes([0, 1/4*3+1/4/3*2.5, 1, 1/4]) #axes[1]
    axes.append(ax)
    ax.plot(x, df_sys[v]/100, c=colors[v], label="Sim", lw=lw_sim, zorder=10)
    for df in stochastic:
        ax.plot(x, df[v]/100, c=colors[v], alpha=0.3, lw=lw_sto, zorder=5)
    if data is not None:
        data_ = data.loc[x, :]
        ax.plot(x, data_[v]/100, c="k", ls="--", label="Obv", lw=lw_obv, zorder=20)
    if metrices is not None:
        ax.text(0.05, 0.05, f"RMSE: {round(metrices.loc[v, 'rmse'], 3)}", transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='left')
    if wet_dry_bg:
        plot_wet_dry_bg(ax)
    #ax.legend(loc="upper right", ncols=2)
    add_vlines(ax)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.set_ylabel("(b)\nWithdrawal\n($10^6$ $m^3$)")
    ax.yaxis.set_label_coords(ylabel_xloc, 0.5)

    #  # Energy
    # v = 'Energy per water use'
    # ax = fig.add_axes([0, 1/6*2+1/6/3*2.5, 1, 1/6]) # axes[2]
    # axes.append(ax)
    # ax.plot(x, df_sys[v], c=colors["Well"], label="Sim", lw=lw_sim, zorder=10)
    # for df in stochastic:
    #     ax.plot(x, df[v], c=colors["Well"], alpha=0.3, lw=lw_sto, zorder=5)
    # plot_wet_dry_bg(ax)
    # #ax.legend(loc="upper left", ncols=2)
    # add_vlines(ax)
    # ax.set_ylim([4.05, 4.38])
    # ax.set_yticks([4.1,4.2,4.3])
    # ax.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)
    # ax.set_ylabel("(c)\nEnergy per\nirr. depth\n(PJ/cm)")
    # ax.yaxis.set_label_coords(ylabel_xloc, 0.5)

    # # Profit
    # v = 'Profit per water use'
    # ax = fig.add_axes([0, 1/6*1+1/6/3*2.5, 1, 1/6]) #axes[3]
    # axes.append(ax)
    # ax.plot(x, df_sys[v], c=colors["Profit"], label="Sim", lw=lw_sim, zorder=10)
    # for df in stochastic:
    #     ax.plot(x, df[v], c=colors["Profit"], alpha=0.3, lw=lw_sto, zorder=5)
    # if wet_dry_bg:
    #     plot_wet_dry_bg(ax)
    # #ax.legend(loc="upper left", ncols=2)
    # add_vlines(ax)
    # ax.set_ylim([-0.02, 1.17])
    # ax.set_yticks([0,0.4,0.8])
    # ax.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)
    # ax.set_ylabel("(d)\nProfit per\nirr. depth\n($10^4$ $/cm)")
    # ax.yaxis.set_label_coords(ylabel_xloc, 0.5)
     
     # CONSUMAT state ratios
    ax = fig.add_axes([0, 4/7, 1, 1/4+1/4/8]) #axes[3]
    axes.append(ax)
    states = ["Imitation", "Social comparison", "Repetition", "Deliberation"]

    dff = df_sys[states]
    num_agt = sum(dff.iloc[0, :])
    dff = dff/num_agt
    for i, state in enumerate(states):
        ax.plot(x, dff[state], zorder=20, label=state, c=colors[state], lw=lw_sim)

    if stochastic is not None:
        for df in stochastic:
            dff = df[states]
            num_agt = sum(dff.iloc[0, :])
            dff = dff/num_agt
            for i, state in enumerate(states):
                ax.plot(x, dff[state], c=colors[state], alpha=0.3, lw=lw_sto, zorder=5)
    if wet_dry_bg:
        plot_wet_dry_bg(ax)
    add_vlines(ax)
    ax.set_ylim([-0.02, 1])
    ax.set_yticks([0,0.25,0.5,0.75,1])
    ax.set_ylabel("(c)\nCONSUMAT\nstate ratio\n(-)")
    ax.set_xlabel("Year")
    ax.legend(ncols=2, labelspacing=0.1, loc="upper right", bbox_to_anchor=(1, 1.35), frameon=False)
    ax.yaxis.set_label_coords(ylabel_xloc, 0.5)
    #plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, dpi=500, bbox_inches='tight')
    plt.show()

# Plot
plot_main_ts(df_sys, data, None, wet_dry_bg=True)

#%% Fig: Crop ratios
fontsize = 10.8
plt.rcParams['font.size'] = fontsize
def plot_crop_ratio(df_sys, data=None, metrices=None, wet_dry_bg=True, stochastic=[],
                    crop_options=["corn", "sorghum", "soybeans", "wheat", "fallow"], savefig=None):
    def add_vlines(ax):
        ax.axvline(2013, c="red", ls=":", lw=1, zorder=1)
        ax.axvline(2018, c="grey", ls=":", lw=1, zorder=1)
        ax.set_xlim([2012, 2022])

    ##### Crop ratio figwd["1.5"], figwd["1.5"]*1.3
    nrows = len(crop_options)
    abcde = ["a", "b", "c", "d", "e"]
    fig, axes = plt.subplots(ncols=1, nrows=nrows, figsize=(figwd["2"], figwd["2"]*1.2),
                             sharex=True, sharey=True)
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

        ax.plot(x, df_sys[crop], c=colors[crop], zorder=100, lw=2)
        if data is not None:
            data_ = data.loc[x, :]
            ax.plot(x, data_[crop], c="k", ls="--", zorder=1000, lw=1)

        if stochastic is not None:
            for df in stochastic:
                ax.plot(x, df[crop], c=colors[crop], zorder=1, alpha=0.2, lw=0.5)

        ax.set_ylabel(f'({abcde[i]})\n'+crop.capitalize()+" ratio", fontsize=fontsize)
        ax.set_ylim([-0.01, 1])
        add_vlines(ax)
        plot_wet_dry_bg(ax)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Year")
    line_obv = Line2D([0], [0], label='Obv', c='k', ls='--')
    line_sim = Line2D([0], [0], label='Sim', c='k', ls='-')
    rect = Patch(facecolor=colors["dry_year"], edgecolor=None, label='Dry year')

    plt.legend(handles=[line_obv, line_sim, rect], labelspacing=0.8,
               frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.1))

    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig, dpi=500, bbox_inches='tight')
    plt.show()

# Plot
plot_crop_ratio(df_sys, data, None, wet_dry_bg=True)



