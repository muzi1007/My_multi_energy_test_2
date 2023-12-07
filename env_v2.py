import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

import pandapower as ppe
import pandapipes as ppt

from electric_env_v2 import *
from thermal_env_v2 import *
from parameters import *

class env_v2():
    def __init__(self):
        self.electric_net = electric_env_v2(future_style='random_noise', compress_style='PCA', future_scale=1)
        self.thermal_net = thermal_env_v2(future_style='random_noise', compress_style='PCA', future_scale=1)
        self.state_space = len(self.electric_net.state_space_ids) + len(self.thermal_net.state_space_ids)
        self.action_space = len(self.electric_net.action_space_ids) + len(self.thermal_net.action_space_ids)
        self.MtoP = 0.00066
        self.PtoM = 1503
        self.Ptoft3 = 3412.14
        self.battery_soc_percent = 0
        self.mass_storage_mass_percent = 0
        self.electric_buy_price = self.electric_net.electric_buy_price
        self.electric_sell_price = self.electric_net.electric_sell_price
        self.gas_price = self.thermal_net.gas_price
        self.loads = self.electric_net.loads
        self.sinks_m = self.thermal_net.sinks_m
        self.pv_p = self.electric_net.pv_p
        self.wind_p = self.electric_net.wind_p

        self.states_records_df = pd.DataFrame(columns=['electric_price', 'gas_price', 'loads', 'pv_p', 'wind_p', 'battery_soc_percent', 'sinks_m', 'mass_storage_mass_percent'])
        self.actions_records_df = pd.DataFrame(columns=['CHP_p', 'battery_p', 'CHP_m', 'Heat_Pump_m', 'Natural_Gas_Boiler_m', 'mass_storage_m'])

    def reset(self):
        self.electric_net = electric_env_v2(future_style='random_noise', compress_style='PCA')
        self.thermal_net = thermal_env_v2(future_style='random_noise', compress_style='PCA')
        states = [self.electric_buy_price, self.gas_price, self.loads, self.pv_p, self.wind_p, self.battery_soc_percent,
                  self.sinks_m, self.mass_storage_mass_percent]
        return states

    def modify_values(self, battery_p, battery_soc_percent, pv_p, wind_p, CHP_p, loads, mass_storage_m, mass_storage_mass_percent, CHP_thermal_m, Heat_Pump_m, Natural_Gas_Boiler_m, sinks_m):
        self.electric_net.modify_values(battery_p, battery_soc_percent, pv_p, wind_p, CHP_p, loads)
        self.thermal_net.modify_values(mass_storage_m, mass_storage_mass_percent, CHP_thermal_m, Heat_Pump_m, Natural_Gas_Boiler_m, sinks_m)

    def run_flow(self):
        self.electric_net.run_flow()
        self.thermal_net.run_flow()
        self.battery_next_soc_percent = self.electric_net.get_next_soc_percent()
        self.mass_storage_next_mass_percent = self.thermal_net.mass_storage_next_mass_percent()
        return self.battery_next_soc_percent, self.mass_storage_next_mass_percent

    def get_states(self):
        self.electric_ext_grid_purchase = self.electric_net.net.res_ext_grid['p_mw']
        self.thermal_ext_grid_purchase = self.thermal_net.net.res_ext_grid['mdot_kg_per_s']

        self.electric_vm_pu = self.electric_net.net.res_bus['vm_pu']
        self.electric_svc3 = self.electric_net.net.res_svc['q_mvar'][0]
        self.electric_svc6 = self.electric_net.net.res_svc['q_mvar'][1]
        return self.electric_ext_grid_purchase, self.thermal_ext_grid_purchase, self.electric_vm_pu, self.electric_svc3, self.electric_svc6

    def cal_reward(self):
        # cost = electric_price * (electricity_grid + Heat_pump*60*60 * MtoP / 3) + heat_price * (thermal_grid*60*60) + gas_price * ((Natural_gas_boiler*60*60* 0.00066 * 3412.14)+(CHP * 3412.14))
        if self.electric_net.net.res_ext_grid.p_mw[0] > 0: self.cost_electricity_ext_grid = self.electric_buy_price * self.electric_net.net.res_ext_grid.p_mw[0]
        elif self.electric_net.net.res_ext_grid.p_mw[0] < 0: self.cost_electricity_ext_grid = self.electric_sell_price * self.electric_net.net.res_ext_grid.p_mw[0]
        self.cost_Heat_Pump = self.electric_buy_price * self.thermal_net.net.res_source.mdot_kg_per_s[1]*60*60 * self.MtoP / 3
        self.cost_Heat_ext_grid = 0.000035 * self.thermal_net.net.res_ext_grid.mdot_kg_per_s[0] # haven't done the progressive rate yet.
        self.cost_Natural_Gas_Boiler = self.gas_price * self.thermal_net.net.res_source.mdot_kg_per_s[2]*60*60 * self.MtoP * self.Ptoft3
        self.cost_CHP = self.gas_price * self.electric_net.net.res_sgen.p_mw[2] * self.Ptoft3

        return -(self.cost_electricity_ext_grid + self.cost_Heat_Pump + self.cost_Heat_ext_grid + self.cost_Natural_Gas_Boiler + self.cost_CHP)

    def record_step(self, next_states, actions):
        self.states_records_df = self.states_records_df.append(pd.Series(next_states, index=self.states_records_df.columns), ignore_index=True)
        self.actions_records_df = self.actions_records_df.append(pd.Series(actions, index=self.actions_records_df.columns), ignore_index=True)

    def get_records(self):
        return self.states_records_df, self.actions_records_df

    def step(self, time_step, actions):
        # 這裡要改成實際值，buffer存的state和next_state是預測值！
        self.loads = self.electric_net.loads_data.iloc[time_step, 0]
        self.pv_p = self.electric_net.pv_data.iloc[time_step, 0]
        self.wind_p = self.electric_net.wind_data.iloc[time_step, 0]
        self.electric_buy_price = self.electric_net.electricity_buy_price_data.iloc[time_step, 0]
        self.electric_sell_price = self.electric_net.electricity_sell_price_data.iloc[time_step, 0]

        self.sinks_m = self.thermal_net.sinks_m_data.iloc[time_step, 0]
        self.gas_price = self.thermal_net.gas_price_data.iloc[time_step, 0]

        done = 0

        # p_mw
        self.battery_p = actions[0]
        self.CHP_p = actions[1]

        # mdot_kg_per_s
        self.CHP_m = actions[1] * self.PtoM
        self.Heat_Pump_m = actions[2]
        self.Natural_Gas_Boiler_m = actions[3]
        self.mass_storage_m = actions[4]

        self.modify_values(self.battery_p, self.battery_soc_percent, self.pv_p, self.wind_p, self.CHP_p, self.loads, self.mass_storage_m, self.mass_storage_mass_percent, self.CHP_m, self.Heat_Pump_m, self.Natural_Gas_Boiler_m, self.sinks_m)

        self.battery_next_soc_percent, self.mass_storage_next_mass_percent = self.run_flow()

        next_states = [self.electric_net.electricity_buy_price_data.iloc[time_step+1, 0],
                       self.thermal_net.gas_price_data.iloc[time_step+1, 0],
                       self.electric_net.loads_data.iloc[time_step+1, 0],
                       self.electric_net.pv_data.iloc[time_step+1, 0],
                       self.electric_net.wind_data.iloc[time_step+1, 0],
                       self.battery_next_soc_percent,
                       self.thermal_net.sinks_m_data.iloc[time_step+1, 0],
                       self.mass_storage_next_mass_percent]

        rewards = self.cal_reward()

        if time_step == len(self.electric_net.loads_data):
            done = 1

        return next_states, rewards, done