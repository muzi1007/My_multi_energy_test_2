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

    def modify_values(self, battery33_p, battery33_soc_percent, pv22_p, wind18_p, CHP_p, loads, mass_storage_m, CHP_thermal_m, Heat_Pump_m, Natural_Gas_Boiler_m, sinks_m):
        self.electric_net.modify_values(battery33_p, battery33_soc_percent, pv22_p, wind18_p, CHP_p, loads)
        self.thermal_net.modify_values(mass_storage_m, CHP_thermal_m, Heat_Pump_m, Natural_Gas_Boiler_m, sinks_m)

    def run_flow(self, mass_storage22_mass_percent):
        self.electric_net.run_flow()
        self.thermal_net.run_flow()
        self.battery33_next_soc_percent = self.electric_net.get_next_soc_percent()
        self.mass_storage22_next_mass_percent = self.thermal_net.mass_storage22_next_mass_percent(mass_storage22_mass_percent)
        return self.battery33_next_soc_percent, self.mass_storage22_next_mass_percent

    def get_states(self):
        self.electric_ext_grid_purchase = self.electric_net.net.res_ext_grid['p_mw']
        self.thermal_ext_grid_purchase = self.thermal_net.net.res_ext_grid['mdot_kg_per_s']

        self.electric_vm_pu = self.electric_net.net.res_bus['vm_pu']
        self.electric_svc3 = self.electric_net.net.res_svc['q_mvar'][0]
        self.electric_svc6 = self.electric_net.net.res_svc['q_mvar'][1]
        return self.electric_ext_grid_purchase, self.thermal_ext_grid_purchase, self.electric_vm_pu, self.electric_svc3, self.electric_svc6
