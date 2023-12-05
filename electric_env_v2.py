import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

import pandapower as ppe

from parameters import *

class electric_env_v2():
    def __init__(self):
        self.net = ppe.networks.case33bw()
        self.state_space_ids = ['loads', 'pv_p', 'wind_p', 'battery_soc_percent', 'electricity_price']
        self.action_space_ids = ['CHP_p', 'battery_p']

        # Calculate cos_phi
        '''
        self.load_cos_phi=[]
        for i in range(len(self.net.load)):
            self.load_cos_phi.append(self.net.load.p_mw[0]/np.sqrt(self.net.load.p_mw[0]**2+self.net.load.q_mvar[0]**2))
        '''
        self.load_cos_phi = 0.8574929257125442

        # Create battery
        self.battery = ppe.create_storage(self.net, 32, p_mw=0.0, max_e_mwh=E_Bat1_E, max_p_mw=E_Bat1_c_max, min_p_mw=E_Bat1_d_min, soc_percent=0.0, name='Electric Battery')
        # Create PV
        self.pv = ppe.create_sgen(self.net, 21, p_mw=0.0, q_mvar=0, name='PV', type='PV')

        # Create Wind turbine
        self.wind = ppe.create_sgen(self.net, 17, p_mw=0.0, q_mvar=0, name='Wind turbine', type='Wind Turbine')

        # Create CHP
        self.CHP = ppe.create_sgen(self.net, 0, p_mw=0.0, q_mvar=0, name='CHP', type='CHP')

        # Create SVCs
        self.svc3 = ppe.create_svc(self.net, 2, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=1, thyristor_firing_angle_degree=90, controllable=True, in_service=True)
        self.svc6 = ppe.create_svc(self.net, 5, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=1, thyristor_firing_angle_degree=90, controllable=True, in_service=True)

    def modify_values(self, battery_p, battery_soc_percent, pv_p, wind_p, CHP_p, loads):
        self.net.storage.at[0, 'p_mw'] = battery_p
        self.net.storage.at[0, 'soc_percent'] = battery_soc_percent
        self.net.sgen.at[0, 'p_mw'] = pv_p
        self.net.sgen.at[1, 'p_mw'] = wind_p
        self.net.sgen.at[2, 'p_mw'] = CHP_p
        for i in range(20):
            self.net.load.at[i, 'p_mw']  = loads[i]
            self.net.load.at[i, 'q_var'] = loads[i] * self.load_cos_phi

    def run_flow(self):
        ppe.runpp(self.net, numba=False)

    def get_storage_energy(self):
        # calculating the stored energy
        self.battery_max_e_mwh = self.net.storage['max_e_mwh']
        self.battery_soc_percent = self.net.storage['soc_percent']
        return self.battery_max_e_mwh * self.battery_soc_percent / 100

    def get_next_soc_percent(self):
        # calculating soc_percent of battery at next step
        self.battery_p_mw = self.net.storage['p_mw']
        self.battery_max_e_mwh = self.net.storage['max_e_mwh']
        self.battery_soc_percent = self.net.storage['soc_percent']
        self.battery_next_soc_percent = self.battery_soc_percent + (self.battery_p_mw * 60 / 60) / self.battery_max_e_mwh
        return self.battery_next_soc_percent

    def simple_plot(self):
        ppe.plotting.simple_plot(self.net, plot_gens=True, plot_sgens=True, plot_line_switches=True, plot_loads=True)

if __name__ == "__main__":
    """
    Functional Test
    """
    electric_net = electric_env_v2()
    electric_net.run_flow()
    electric_net.simple_plot()