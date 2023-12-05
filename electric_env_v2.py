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

        # Calculate cos_phi
        '''
        self.load_cos_phi=[]
        for i in range(len(self.net.load)):
            self.load_cos_phi.append(self.net.load.p_mw[0]/np.sqrt(self.net.load.p_mw[0]**2+self.net.load.q_mvar[0]**2))
        '''
        self.load_cos_phi = 0.8574929257125442

        # Create battery
        self.battery33 = ppe.create_storage(self.net, 32, p_mw=0.0, max_e_mwh=E_Bat1_E, max_p_mw=E_Bat1_c_max, min_p_mw=E_Bat1_d_min, soc_percent=0.0, name='Electric Battery')
        # Create PV
        self.pv22 = ppe.create_sgen(self.net, 21, p_mw=0.0, q_mvar=0, name='PV', type='PV')

        # Create Wind turbine
        self.wind18 = ppe.create_sgen(self.net, 17, p_mw=0.0, q_mvar=0, name='Wind turbine', type='Wind Turbine')

        # Create CHP
        self.CHP1 = ppe.create_sgen(self.net, 0, p_mw=0.0, q_mvar=0, name='CHP', type='CHP')

        # Create SVCs
        self.svc3 = ppe.create_svc(self.net, 2, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=1, thyristor_firing_angle_degree=90, controllable=True, in_service=True)
        self.svc6 = ppe.create_svc(self.net, 5, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=1, thyristor_firing_angle_degree=90, controllable=True, in_service=True)

    def modify_values(self, battery33_p, battery33_soc_percent, pv22_p, wind18_p, CHP_p, loads):
        self.net.storage.at[0, 'p_mw'] = battery33_p
        self.net.storage.at[0, 'soc_percent'] = battery33_soc_percent
        self.net.sgen.at[0, 'p_mw'] = pv22_p
        self.net.sgen.at[1, 'p_mw'] = wind18_p
        self.net.sgen.at[2, 'p_mw'] = CHP_p
        for i in range(20):
            self.net.load.at[i, 'p_mw']  = loads[i]
            self.net.load.at[i, 'q_var'] = loads[i] * self.load_cos_phi

    def run_flow(self):
        ppe.runpp(self.net, numba=False)

    def get_storage_energy(self):
        # calculating the stored energy
        self.battery33_max_e_mwh = self.net.storage['max_e_mwh']
        self.battery33_soc_percent = self.net.storage['soc_percent']
        return self.battery33_max_e_mwh * self.battery33_soc_percent / 100

    def get_next_soc_percent(self):
        # calculating soc_percent of battery at next step
        self.battery33_p_mw = self.net.storage['p_mw']
        self.battery33_max_e_mwh = self.net.storage['max_e_mwh']
        self.battery33_soc_percent = self.net.storage['soc_percent']
        self.battery33_next_soc_percent = self.battery33_soc_percent + (self.battery33_p_mw * 60 / 60) / self.battery33_max_e_mwh
        return self.battery33_next_soc_percent

if __name__ == "__main__":
    """
    Functional Test
    """
    electric_net = electric_env_v2()
    ppe.runpp(electric_net.net, numba=False)
    ppe.plotting.simple_plot(electric_net.net, plot_gens=True, plot_sgens=True, plot_line_switches=True, plot_loads=True)