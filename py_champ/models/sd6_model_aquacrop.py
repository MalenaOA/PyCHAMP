from copy import deepcopy

import gurobipy as gp
import mesa
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from ..components.aquifer import Aquifer
from ..components.behavior import Behavior4Aquacrop
# from ..components.field import Field4SingleFieldAndWell
from ..components.field import Field_aquacrop
from ..components.finance import Finance4SingleFieldAndWell
from ..components.optimization_1f1w import Optimization4SingleFieldAndWell
from ..components.well import Well4SingleFieldAndWell
from ..utility.util import (
    BaseSchedulerByTypeFiltered,
    Indicator,
    TimeRecorder,
    get_agt_attr,
)


# % MESA
class SD6ModelAquacrop(mesa.Model):
    """ Based on the SD6Model_1f1w class but using aquacrop to calculate yield."""

    def __init__(
        self,
        pars,
        crop_options,
        prec_aw_step,
        aquifers_dict,
        fields_dict,
        wells_dict,
        finances_dict,
        behaviors_dict,
        components=None,
        optimization_class=Optimization4SingleFieldAndWell,
        init_year=2011,
        end_year=2022,
        lema_options=(True, "wr_LEMA_5yr", 2013),
        fix_state=None,
        show_step=True,
        show_initialization=True,
        seed=None,
        gurobi_dict=None,
        **kwargs,
    ):
        # Time and Step recorder
        self.time_recorder = TimeRecorder()

        # Prepare the model
        if gurobi_dict is None:
            gurobi_dict = {"LogToConsole": 0, "NonConvex": 2, "Presolve": -1}
        if components is None:
            components = {
                "aquifer": Aquifer,
                "field": Field_aquacrop,
                "well": Well4SingleFieldAndWell,
                "finance": Finance4SingleFieldAndWell,
                "behavior": Behavior4Aquacrop,
            }
        self.components = components
        self.optimization_class = optimization_class
        self.crop_options = crop_options  # n_c

        # Initialize the model
        self.init_year = init_year
        self.start_year = self.init_year + 1
        self.end_year = end_year
        self.total_steps = self.end_year - self.init_year
        self.current_year = self.init_year
        self.t = 0  # This is initial step

        # LEMA settings
        self.lema = lema_options[0]
        self.lema_wr_name = lema_options[1]
        self.lema_year = lema_options[2]
        self.show_step = show_step

        # mesa has self.random.seed(seed) but it is not usable for truncnorm
        # We create our own random generator.
        self.seed = seed
        self.rngen = np.random.default_rng(seed)

        # Store parameters
        self.pars = pars

        # Input timestep data
        self.prec_aw_step = prec_aw_step  # [prec_aw_id][year][crop]
        self.crop_price_step = kwargs.get("crop_price_step")  # [finance_id][year][crop]

        # Create schedule and assign it to the model
        self.schedule = BaseSchedulerByTypeFiltered(self)

        # Copy the dictionaries before correction from shared_config
        aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict = (
            deepcopy(aquifers_dict),
            deepcopy(fields_dict),
            deepcopy(wells_dict),
            deepcopy(finances_dict),
            deepcopy(behaviors_dict),
        )

        # Setup Gurobi environment, gpenv
        self.gpenv = gp.Env(empty=True)
        for k, v in gurobi_dict.items():
            self.gpenv.setParam(k, v)
        self.gpenv.start()

        # Initialize aquifer agent (this is not associated with farmers)
        aquifers = {}
        for aqid, aquifer_dict in aquifers_dict.items():
            agt_aquifer = self.components["aquifer"](
                unique_id=aqid, model=self, settings=aquifer_dict
            )
            aquifers[aqid] = agt_aquifer
            self.schedule.add(agt_aquifer)
        self.aquifers = aquifers

        # Initialize field agents
        fields = {}
        for fid, field_dict in fields_dict.items():
            # # Initialize crop type
            # if isinstance(field_dict["init"]["crop"], list):
            #     raise NotImplementedError(
            #         "Multiple crop types per field is not supported. "
            #         +"Initial crop type must be a single string."
            #         )

            # Initialize fields
            agt_field = self.components["field"](
                unique_id=fid,
                model=self,
                settings=field_dict,
                # For this model's convenience
                irr_freq=field_dict.get("irr_freq"),
                truncated_normal_pars=field_dict.get("truncated_normal_pars"),
                lat=field_dict.get("lat"),
                lon=field_dict.get("lon"),
                x=field_dict.get("x"),
                y=field_dict.get("y"),
            )
            fields[fid] = agt_field
            self.schedule.add(agt_field)
        self.fields = fields

        # Initialize wells
        wells = {}
        for wid, well_dict in wells_dict.items():
            agt_well = self.components["well"](
                unique_id=wid, model=self, settings=well_dict
                )
            wells[wid] = agt_well
            self.schedule.add(agt_well)
        self.wells = wells

        # Initialize behavior and finance agents and append them to the schedule
        ## Don't use parallelization. It is slower!

        behaviors = {}
        finances = {}
        for behavior_id, behavior_dict in tqdm(
            behaviors_dict.items(),
            desc="Initialize behavior agents",
            disable=not show_initialization
        ):
            # Initialize finance
            finance_id = behavior_dict["finance_id"]
            finance_dict = finances_dict[finance_id]
            agt_finance = self.components["finance"](
                unique_id=f"{finance_id}_{behavior_id}",
                model=self,
                settings=finance_dict,
            )
            agt_finance.finance_id = finance_id
            # Assume one behavior agent has one finance object
            finances[behavior_id] = agt_finance  
            self.schedule.add(agt_finance)

            agt_behavior = self.components["behavior"](
                unique_id=behavior_id,
                model=self,
                settings=behavior_dict,
                pars=self.pars,
                fields={
                    fid: self.fields[fid]
                    for i, fid in enumerate(behavior_dict["field_ids"])
                },
                wells={
                    wid: self.wells[wid]
                    for i, wid in enumerate(behavior_dict["well_ids"])
                },
                finance=agt_finance,
                aquifers=self.aquifers,
                optimization_class=self.optimization_class,
                # kwargs
                fix_state=fix_state,
            )
            behaviors[behavior_id] = agt_behavior
            self.schedule.add(agt_behavior)
        self.behaviors = behaviors
        self.finances = finances

        # Update the crop price for the initial year
        if self.crop_price_step is not None:
            for _, finance in self.finances.items():
                crop_prices = self.crop_price_step.get(finance.finance_id)
                if crop_prices is not None:
                    finance.crop_price = crop_prices[self.current_year]

        # Data collector
        agent_reporters = {
            "agt_type": get_agt_attr("agt_type"),
            ### Field
            "field_type": get_agt_attr("field_type"),
            "crop": get_agt_attr("crop"),
            "irr_vol": get_agt_attr("irr_vol_per_field"),  # m-ha
            "yield_rate": get_agt_attr("yield_rate_per_field"),
            "yield": get_agt_attr("y"),  # 1e4 bu/field
            "w": get_agt_attr("w"),
            "field_area": get_agt_attr("field_area"),  # ha
            ### Behavior
            "Sa": get_agt_attr("satisfaction"),
            "E[Sa]": get_agt_attr("expected_sa"),
            "Un": get_agt_attr("uncertainty"),
            "state": get_agt_attr("state"),
            "gp_status": get_agt_attr("gp_status"),
            "gp_MIPGap": get_agt_attr("gp_MIPGap"),
            ### Finance
            "profit": get_agt_attr("finance.profit"),  # 1e4 $
            "revenue": get_agt_attr("finance.rev"),  # 1e4 $
            "energy_cost": get_agt_attr("finance.cost_energy"),  # 1e4 $
            "tech_cost": get_agt_attr("finance.cost_tech"),  # 1e4 $
            ### Well
            "water_depth": get_agt_attr("l_wt"),
            "energy": get_agt_attr("e"),  # PJ
            ### Aquifer
            "withdrawal": get_agt_attr("withdrawal"),  # m-ha
            "GW_st": get_agt_attr("st"),  # m
            "GW_dwl": get_agt_attr("dwl"),  # m
        }
        model_reporters = {}
        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
        )

        # Summary info
        estimated_sim_dur = self.time_recorder.sec2str(
            self.time_recorder.get_elapsed_time(strf=False) * self.total_steps
        )
        msg = f"""\n
        Initial year: \t{self.init_year}
        Simulation period:\t{self.start_year} to {self.end_year}
        Number of agents:\t{len(behaviors_dict)}
        Number of aquifers:\t{len(aquifers_dict)}
        Initialiation duration:\t{self.time_recorder.get_elapsed_time()}
        Estimated sim duration:\t{estimated_sim_dur}
        """
        if show_initialization:
            print(msg)

        # self.csv_path = "/Users/michellenguyen/Downloads/PyCHAMP/examples/Aquacrop/data/corn_default.csv"
        self.csv_path = "C:\\Users\\m154o020\\CHAMP\\PyCHAMP\\Summer2024\\code_20240705\\PyCHAMP\\examples\\Aquacrop\\data\\corn_default.csv"
        #self.csv_path = "D:\Malena\CHAMP\PyCHAMP\code_20240704\PyCHAMP\examples\Aquacrop\corn_default.csv"

    def step(self):
        """
        Advance the model by one step.

        This method progresses the simulation by one year. It involves updating crop 
        prices, deciding field types based on agent behavior, applying LEMA policy, and 
        executing the step functions of all agents. Finally, it collects data and 
        updates the model's state.

        The method controls the flow of the simulation, ensuring that each agent and 
        component of the model acts according to the current time step and the state of
        the environment.
        """
        self.current_year += 1
        self.t += 1

        ##### Human (behaviors)
        current_year = self.current_year

        # Update crop price
        if self.crop_price_step is not None:
            for _, finance in self.finances.items():
                crop_prices = self.crop_price_step.get(finance.finance_id)
                finance.crop_price = crop_prices[current_year]

        # Assign field type based on each behavioral agent.
        for _, behavior in self.behaviors.items():
            # Randomly select rainfed field
            for _, field in behavior.fields.items():
                rn_irr = True
                if rn_irr:
                    irr_freq = field.irr_freq
                    rn = self.rngen.uniform(0, 1)
                    if rn <= irr_freq:
                        field.field_type = "optimize"  # Optimize it
                        field.field_type_rn = "optimize"
                    else:
                        # raise
                        field.field_type = "rainfed"  # Force the field to be rainfed
                        field.field_type_rn = "rainfed"

            # Turn on LEMA (i.e., a water right constraint) starting from 2013
            if self.lema and current_year >= self.lema_year:
                behavior.wr_dict[self.lema_wr_name]["status"] = True
            else:
                behavior.wr_dict[self.lema_wr_name]["status"] = False

            # Save the decisions from the previous step. 
            # (very important for retrieving the neighbor's previous decision)
            behavior.pre_dm_sols = behavior.dm_sols

        # Simulation
        # Exercute step() of all behavioral agents in a for loop
        # Note: fields, wells, and finance are simulation within a behavioral agent to better accommodate heterogeneity among behavioral agents
        self.schedule.step(agt_type="Behavior")  # Parallelization makes it slower!

        ##### Nature Environment (aquifers)
        for aq_id, aquifer in self.aquifers.items():
            withdrawal = 0.0
            # Collect all the well withdrawals of a given aquifer
            withdrawal += sum(
                [
                    well.withdrawal if well.aquifer_id == aq_id else 0.0
                    for _, well in self.wells.items()
                ]
            )
            # Update aquifer
            aquifer.step(withdrawal)

        # Collect df_sys and print info
        self.datacollector.collect(self)
        if self.lema and current_year == self.lema_year and self.show_step:
            print("LEMA begin")
        if self.show_step:
            print(
                f"Year {self.current_year} [{self.t}/{self.total_steps}]"
                + f"\t{self.time_recorder.get_elapsed_time()}\n"
            )

        # Termination criteria
        if self.current_year == self.end_year:
            self.running = False
            print("Done!", f"\t{self.time_recorder.get_elapsed_time()}")

        # # Looping logic
        # for field_id, field in self.fields.items():
        #     irr_depth, crop, prec_aw = field.get_data_for_aquacrop()
        #     for i in range(self.total_steps):
        #         y, avg_y_y, irr_vol, crop, bias_corrected_yield, bias_corrected_irrigation, irr_depth = field.step(irr_depth, field.i_crop, prec_aw, self.csv_path)
        #         print(f"Step {i+1}/{self.total_steps}: Yield={y}, Irrigation Volume={irr_vol}, Bias-corrected Yield={bias_corrected_yield}, Bias-corrected Irrigation={bias_corrected_irrigation}")
        #         field.update_field_data(bias_corrected_yield, bias_corrected_irrigation)

# Looping logic
        for field_id, field in self.fields.items():
            for i in range(self.total_steps):
        # Run the step method, which already handles whether to use AquaCrop or the default simulation
                y, avg_y_y, irr_vol, crop, bias_corrected_yield, bias_corrected_irrigation, irr_depth = field.step(
                field.irr_depth, 
                field.i_crop, 
                field.prec_aw, 
                self.csv_path
                )
        
        # Print the results for this step
            print(f"Step {i+1}/{self.total_steps}: Crop={crop}, Yield={y}, Irrigation Volume={irr_vol}, Bias-corrected Yield={bias_corrected_yield}, Bias-corrected Irrigation={bias_corrected_irrigation}")
        
        # Update field data if necessary
        field.update_field_data(bias_corrected_yield, bias_corrected_irrigation)


    
    def end(self):
        """Depose the Gurobi environment, ensuring that it is executed only when
        the instance is no longer needed.
        """
        self.gpenv.dispose()

    @staticmethod
    def get_dfs(model):
        df = model.datacollector.get_agent_vars_dataframe().reset_index()
        df["year"] = df["Step"] + model.init_year
        df.index = df["year"] #(-)

        # =============================================================================
        # df_agt
        # =============================================================================
        df_behaviors = df[df["agt_type"] == "Behavior"].dropna(axis=1, how="all")
        df_behaviors["bid"] = df_behaviors["AgentID"]

        df_fields = df[df["agt_type"] == "Field"].dropna(axis=1, how="all")
        df_fields["field_type"] = ""
        df_fields.loc[df_fields["irr_vol"] == 0, "field_type"] = "rainfed"
        df_fields.loc[df_fields["irr_vol"] > 0, "field_type"] = "irrigated"
        df_fields["irr_depth"] = (df_fields["irr_vol"] / df_fields["field_area"] * 100)  # cm
        df_fields["irr_vol_m3"] = df_fields["irr_vol"]/100 #m3
        df_fields["fid"] = df_fields["AgentID"]

        df_wells = df[df["agt_type"] == "Well"].dropna(axis=1, how="all")
        df_wells["wid"] = df_wells["AgentID"]

        df_agt = pd.concat(
            [
                df_behaviors[
                    [
                        "bid",
                        "Sa",
                        "E[Sa]",
                        "Un",
                        "state",
                        "profit",
                        "revenue",
                        "energy_cost",
                        "tech_cost",
                        "gp_status",
                        "gp_MIPGap",
                    ]
                ],
                df_fields[
                    [
                        "fid",
                        "field_type",
                        "crop",
                        "irr_vol",
                        "irr_vol_m3",
                        "yield",
                        "yield_rate",
                        "irr_depth",
                        "w",
                        "field_area",
                    ]
                ],
                df_wells[["wid",
                          "water_depth",
                          "withdrawal",
                          "energy"]],
            ],
            axis=1,
        )
        df_agt["yield"] = df_agt["yield"].apply(sum).apply(sum)

        df_agt = df_agt.round(8)

        # =============================================================================
        # df_sys
        # =============================================================================
        df_sys = pd.DataFrame()

        # Saturated Thickness
        df_aquifers = df[df["agt_type"] == "Aquifer"].dropna(axis=1, how="all")
        df_sys["GW_st"] = df_aquifers["GW_st"]

        # Water use
        df_sys["withdrawal"] = df_aquifers["withdrawal"]/100

        # Field_Type ratio
        dff = (
            df_agt[["field_type", "field_area"]]
            .groupby([df_agt.index, "field_type"])
            .sum()
        )
        all_years = dff.index.get_level_values("year").unique()
        all_field_types = ["irrigated", "rainfed"]
        new_index = pd.MultiIndex.from_product(
            [all_years, all_field_types], names=["year", "field_type"]
        )
        dff = dff.reindex(new_index).fillna(0)

        df_sys["rainfed"] = (
                dff.xs("rainfed", level="field_type") / dff.groupby("year").sum()
        )

        df_sys["irrigated"] = (
                dff.xs("irrigated", level="field_type") / dff.groupby("year").sum()
        )

        # Crop ratio
        dff = df_agt[["crop", "field_area"]].groupby([df_agt.index, "crop"]).sum()
        all_years = dff.index.get_level_values("year").unique()
        all_crop_types = model.crop_options
        new_index = pd.MultiIndex.from_product(
            [all_years, all_crop_types], names=["year", "crop"]
        )
        dff = dff.reindex(new_index).fillna(0)
        total = dff.groupby("year").sum()
        for c in all_crop_types:
            df_sys[f"{c}"] = dff.xs(c, level="crop") / total

        # Yield per crop
        dff = df_agt[["crop", "yield"]].groupby([df_agt.index, "crop"]).sum()
        all_years = dff.index.get_level_values("year").unique()
        all_crop_types = model.crop_options
        new_index = pd.MultiIndex.from_product(
            [all_years, all_crop_types], names=["year", "crop"]
        )
        dff = dff.reindex(new_index).fillna(0)
        for c in all_crop_types:
            df_sys[f"{c}_yield"] = dff.xs(c, level="crop")

        # Water use per crop
        dff = df_agt[["crop", "irr_vol_m3"]].groupby([df_agt.index, "crop"]).sum()
        all_years = dff.index.get_level_values("year").unique()
        all_crop_types = model.crop_options
        new_index = pd.MultiIndex.from_product(
            [all_years, all_crop_types], names=["year", "crop"]
        )
        dff = dff.reindex(new_index).fillna(0)
        for c in all_crop_types:
            df_sys[f"{c}_irr_vol"] = dff.xs(c, level="crop")

        # Total irrigation volume
        df_sys["total_irr_vol"] = df_agt.groupby("year")["irr_vol_m3"].sum()

        # Total irrigation depth
        df_sys["total_irr_depth"] = df_agt.groupby("year")["irr_depth"].sum()

        # CONSUMAT state ratio
        dff = df_behaviors[["state"]].groupby([df_behaviors.index, "state"]).size()
        all_years = dff.index.get_level_values("year").unique()
        all_states = ["Imitation", "Social comparison", "Repetition", "Deliberation"]
        new_index = pd.MultiIndex.from_product(
            [all_years, all_states], names=["year", "state"]
        )
        dff = dff.reindex(new_index).fillna(0)

        for s in all_states:
            df_sys[f"{s}"] = dff.xs(s, level="state")

        # Total profit
        df_sys["total_profit"] = df_agt.groupby("year")["profit"].sum()

        # Profit per crop
        dff = df_agt[["crop", "profit"]].groupby([df_agt.index, "crop"]).sum()
        all_years = dff.index.get_level_values("year").unique()
        all_crop_types = model.crop_options
        new_index = pd.MultiIndex.from_product(
            [all_years, all_crop_types], names=["year", "crop"]
        )
        dff = dff.reindex(new_index).fillna(0)
        for c in all_crop_types:
            df_sys[f"{c}_profit"] = dff.xs(c, level="crop")

        # Profit per irrigation depth
        df_sys["profit_irr_depth"] = df_sys["total_profit"] / df_sys["total_irr_depth"]

        # Total energy
        df_sys["total_energy"] = df_agt.groupby("year")["energy"].sum()

        # Energy per irrigation depth
        df_sys["energy_irr_depth"] = df_sys["total_energy"] / df_sys["total_irr_depth"]

        # df_sys = df_sys.round(4)

        # =============================================================================
        # df_aqua
        # =============================================================================

        # Extract the relevant columns from df_agt
        df_aqua = df_agt[["bid", "irr_depth", "crop", "field_type", "yield"]].copy()

        # Rename columns
        df_aqua.rename(columns={"field_type": "irrig_method"}, inplace=True)
        df_aqua.rename(columns={"irr_depth": "irr_depth_pychamp"}, inplace=True)
        df_aqua.rename(columns={"yield": "yield_pychamp_bu"}, inplace=True)

        df_aqua["irrig_method"] = df_aqua["irrig_method"].map({
            "irrigated": 1,
            "rainfed": 0,
            })

        # Create df_aqua_units based on df_aqua
        df_aqua_units = df_aqua.copy()

        # Rename the column
        df_aqua_units.rename(columns={"irr_depth_pychamp": "maxirr_season"}, inplace=True)
        #df_aqua.rename(columns={"yield": "yield_pychamp_ton"}, inplace=True)

        # Convert from cm to mm (1 cm = 10 mm)
        df_aqua_units["maxirr_season"] = df_aqua_units["maxirr_season"] * 10
        #df_aqua_units["yield_pychamp_ton"] = df_aqua_units["yield_pychamp_ton"] / .0254

        return df_sys, df_agt, df_aqua, df_aqua_units
    

    @staticmethod
    def get_metrices(df_sys,
                     data,
                     targets=None,
                     indicators_list=None,
    ):
        """
        Calculate various metrics based on system-level data and specified targets.

        Parameters
        ----------
        df_sys : pd.DataFrame
            The system-level dataframe.
        data : dict or pd.DataFrame
            The reference or observed data for comparison.
        targets : list
            List of targets or variables for which metrics are calculated.
        indicators_list : list
            List of indicators to calculate, such as 'r', 'rmse', 'kge'.

        Returns
        -------
        pd.DataFrame
            A dataframe containing calculated metrics for each target.

        This method is useful for evaluating the performance of the model against
        real-world data or specific objectives, providing insights into the accuracy
        and reliability of the simulation.
        """
        if targets is None:
            targets = [
                "GW_st",
                "withdrawal",
                "rainfed",
                "corn",
                "others"
            ]
        if indicators_list is None:
            indicators_list = ["r", "rmse", "kge"]
        indicators = Indicator()
        metrices = []
        for tar in targets:
            metrices.append(
                indicators.cal_indicator_df(
                    x_obv=data[tar],
                    y_sim=df_sys[tar],
                    index_name=tar,
                    indicators_list=indicators_list,
                )
            )
        metrices = pd.concat(metrices)
        return metrices
        