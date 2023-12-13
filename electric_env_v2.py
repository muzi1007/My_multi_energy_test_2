import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

import pandapower as ppe

from parameters import *

class electric_env_v2():
    def __init__(self, future_style='random_noise', compress_style='PCA', future_scale=1, test=0):

        if test == 1: self.path_prefix = '.'
        else: self.path_prefix = '..'

        self.net = ppe.networks.case33bw()
        false_some_loads_test = 0
        if false_some_loads_test == 1:
            for i in range(18):
                self.net.load.at[i, 'in_service'] = False

        self.state_space_ids = ['loads', 'pv_p', 'wind_p', 'battery_soc_percent', 'electricity_price']
        self.action_space_ids = ['CHP_p', 'battery_p']
        self.future_style = future_style
        self.compress_style = compress_style
        self.future_scale = future_scale
        self.time_step = 0
        if self.future_style == 'random_noise' and self.compress_style == 'PCA':
            self.compress_loads_data = pd.read_csv(f'{self.path_prefix}/state_compress/randomized_{self.future_scale}_reduced_states/load_randomized_{self.future_scale}_reduced_states.csv', usecols=['reduced_states'])
            self.compress_pv_data = pd.read_csv(f'{self.path_prefix}/state_compress/randomized_{self.future_scale}_reduced_states/PV_electricity_randomized_{self.future_scale}_reduced_states.csv', usecols=['reduced_states'])
            self.compress_wind_data = pd.read_csv(f'{self.path_prefix}/state_compress/randomized_{self.future_scale}_reduced_states/Wind_electricity_randomized_{self.future_scale}_reduced_states.csv', usecols=['reduced_states'])
            self.compress_electricity_buy_price_data = pd.read_csv(f'{self.path_prefix}/state_compress/randomized_{self.future_scale}_reduced_states/Electricity_price_buy_price_randomized_{self.future_scale}_reduced_states.csv', usecols=['reduced_states'])
            self.compress_electricity_sell_price_data = pd.read_csv(f'{self.path_prefix}/state_compress/randomized_{self.future_scale}_reduced_states/Electricity_price_sell_price_randomized_{self.future_scale}_reduced_states.csv', usecols=['reduced_states'])
        elif self.future_style == 'LSTM_predict' and self.compress_style == 'PCA':
            self.compress_loads_data = pd.read_csv(f'{self.path_prefix}/state_compress/LSTM_predict_{self.future_scale}_reduced_states/load_predict_{self.future_scale}_reduced_states.csv', usecols=['reduced_states'])
            self.compress_pv_data = pd.read_csv(f'{self.path_prefix}/state_compress/LSTM_predict_{self.future_scale}_reduced_states/PV_electricity_predict_{self.future_scale}_reduced_states.csv', usecols=['reduced_states'])
            self.compress_wind_data = pd.read_csv(f'{self.path_prefix}/state_compress/LSTM_predict_{self.future_scale}_reduced_states/Wind_electricity_predict_{self.future_scale}_reduced_states.csv', usecols=['reduced_states'])
            self.compress_electricity_buy_price_data = pd.read_csv(f'{self.path_prefix}/state_compress/LSTM_predict_{self.future_scale}_reduced_states/Electricity_price_buy_price_predict_{self.future_scale}_reduced_states.csv', usecols=['reduced_states'])
            self.compress_electricity_sell_price_data = pd.read_csv(f'{self.path_prefix}/state_compress/LSTM_predict_{self.future_scale}_reduced_states/Electricity_price_sell_price_predict_{self.future_scale}_reduced_states.csv', usecols=['reduced_states'])

        self.compress_loads = self.compress_loads_data.iloc[0, 0]
        self.compress_pv_p = self.compress_pv_data.iloc[0, 0]
        self.compress_wind_p = self.compress_wind_data.iloc[0, 0]
        self.compress_electric_buy_price = self.compress_electricity_buy_price_data.iloc[0, 0]
        self.compress_electric_sell_price = self.compress_electricity_sell_price_data.iloc[0, 0]

        self.loads = pd.read_csv(f'{self.path_prefix}/data/profile/load_profile.csv')
        self.pv_p = pd.read_csv(f'{self.path_prefix}/data/profile/PV_profile.csv', usecols=['electricity']) # ?
        self.wind_p = pd.read_csv(f'{self.path_prefix}/data/profile/Wind_profile.csv', usecols=['electricity']) # ?
        self.electric_buy_price = pd.read_csv(f'{self.path_prefix}/data/profile/electricity_price_profile.csv', usecols=['buy_price'])  * 963.91
        self.electric_sell_price = pd.read_csv(f'{self.path_prefix}/data/profile/electricity_price_profile.csv', usecols=['sell_price'])  * 963.91

        # Calculate cos_phi
        '''
        self.load_cos_phi=[]
        for i in range(len(self.net.load)):
            self.load_cos_phi.append(self.net.load.p_mw[0]/np.sqrt(self.net.load.p_mw[0]**2+self.net.load.q_mvar[0]**2))
        '''
        self.load_cos_phi = 0.8574929257125442

        # Create battery
        self.E_Bat1_E = E_Bat1_E
        self.battery = ppe.create_storage(self.net, 32, p_mw=0.0, max_e_mwh=E_Bat1_E, max_p_mw=E_Bat1_c_max, min_p_mw=E_Bat1_d_min, soc_percent=0.0, name='Electric Battery', in_service=True)
        # Create PV
        self.pv = ppe.create_sgen(self.net, 21, p_mw=0.0, q_mvar=0, name='PV', type='PV')

        # Create Wind turbine
        self.wind = ppe.create_sgen(self.net, 17, p_mw=0.0, q_mvar=0, name='Wind turbine', type='Wind Turbine')

        # Create CHP
        self.CHP = ppe.create_sgen(self.net, 0, p_mw=0.0, q_mvar=0, name='CHP', type='CHP')

        # Create SVCs
        self.svc3 = ppe.create_svc(self.net, 2, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=1, thyristor_firing_angle_degree=90, controllable=True, in_service=True)
        self.svc6 = ppe.create_svc(self.net, 5, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=1, thyristor_firing_angle_degree=90, controllable=True, in_service=True)

        # Delete default loads
        for i in [0,4,6,11,15,18,19,20,23,25,27,31]:
            self.net['load'].drop(i, inplace=True)

    def modify_values(self, battery_p, battery_soc_percent, pv_p, wind_p, CHP_p, loads):
        self.net.storage.at[0, 'p_mw'] = battery_p
        #print('battery_p: ', battery_p)
        self.net.storage.at[0, 'soc_percent'] = battery_soc_percent
        #print('battery_soc_percent: ', battery_soc_percent)
        self.net.sgen.at[0, 'p_mw'] = pv_p
        #print('pv_p: ', pv_p)
        self.net.sgen.at[1, 'p_mw'] = wind_p
        #print('wind_p: ', wind_p)
        self.net.sgen.at[2, 'p_mw'] = CHP_p
        #print('CHP_p: ', CHP_p)
        load_index = [1,2,3,5,7,8,9,10,12,13,14,16,17,21,22,24,26,28,29,30]
        for i in range(20):
            self.net.load.at[load_index[i], 'p_mw']  = loads[i]
            self.net.load.at[load_index[i], 'q_mvar'] = loads[i] * self.load_cos_phi

    def run_flow(self):
        ppe.runpp(self.net, numba=False, max_iteration=30)

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
        self.battery_next_soc_percent = self.battery_soc_percent + self.battery_p_mw / self.battery_max_e_mwh * 100
        return self.battery_next_soc_percent[0]

    def simple_plot(self):
        ppe.plotting.simple_plot(self.net, plot_gens=True, plot_sgens=True, plot_line_switches=True, plot_loads=True)

if __name__ == "__main__":
    """
    Functional Test
    """
    electric_net = electric_env_v2(future_style='random_noise', compress_style='PCA', future_scale=1, test=1)
    time_step = 6
    battery_p = -1.0783708
    battery_soc_percent = 33.157697319984436
    pv_p = electric_net.pv_p.iloc[time_step, 0]
    wind_p = electric_net.wind_p.iloc[time_step, 0]
    CHP_p = 0.044863224
    loads = electric_net.loads.loc[time_step,:].values[1:]/10
    electric_net.modify_values(battery_p, battery_soc_percent, pv_p, wind_p, CHP_p, loads)
    electric_net.run_flow()
    battery_next_soc_percent = electric_net.get_next_soc_percent()
    print('storage: \n', electric_net.net.storage)
    print('battery_next_soc_percent: ', battery_next_soc_percent)
    electric_net.simple_plot()