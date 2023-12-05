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
        self.electric_net = electric_env_v2()
        self.thermal_net = thermal_env_v2()
        self.state_space = len(self.electric_net.state_space_ids) + len(self.thermal_net.state_space_ids)
        self.action_space = len(self.electric_net.action_space_ids) + len(self.thermal_net.action_space_ids)
        self.records = pd.DataFrame

    def reset(self):
        self.electric_net = electric_env_v2()
        self.thermal_net = thermal_env_v2()

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
        self.cost_electricity_ext_grid = self.electric_price * electric_net.net.res_ext_grid.p_mw[0]
        self.cost_Heat_Pump = self.electric_price * thermal_net.net.res_source.mdot_kg_per_s[1]*60*60 * 0.000665 / 3
        self.cost_Heat_ext_grid = 0.000035 * thermal_net.net.res_ext_grid.mdot_kg_per_s[0] # haven't done the progressive rate yet.
        self.cost_Natural_Gas_Boiler = self.gas_price * thermal_net.net.res_source.mdot_kg_per_s[2]*60*60 *  0.00066 * 3412.14
        self.cost_CHP = self.gas_price * electric_net.net.res_sgen.p_mw[2] * 3412.14

        return -(self.cost_electricity_ext_grid + self.cost_Heat_Pump + self.cost_Heat_grid + self.cost_Natural_Gas_Boiler + self.cost_CHP)

    def step(self, states, actions):
        self.loads = states[0]
        self.pv_p = states[1]
        self.wind_p = states[2]
        self.battery_soc_percent = states[3]
        self.electric_price = states[4]

        self.sinks_m = states[5]
        self.mass_storage_mass_percent = states[6]
        self.gas_price = states[7]

        # p_mw
        self.battery_p = actions[0]
        self.CHP_p = actions[1]

        # mdot_kg_per_s
        self.CHP_m = actions[1] * 1503
        self.Heat_Pump_m = actions[2]
        self.Natural_Gas_Boiler_m = actions[3]
        self.mass_storage_m = actions[4]

        self.modify_values(self.battery_p, self.battery_soc_percent, self.pv_p, self.wind_p, self.CHP_p, self.loads, self.mass_storage_m, self.mass_storage_mass_percent, self.CHP_m, self.Heat_Pump_m, self.Natural_Gas_Boiler_m, self.sinks_m)

        self.battery_next_soc_percent, self.mass_storage_next_mass_percent = self.run_flow()

        next_states = [self.battery_next_soc_percent, self.mass_storage_next_mass_percent]
        rewards = self.cal_reward()

        return next_states, rewards


