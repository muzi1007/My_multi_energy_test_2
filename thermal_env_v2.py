import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

import pandapipes as ppt

from parameters import *

class thermal_env_v2():
    def __init__(self, future_style='random_noise', compress_style='PCA', future_scale=1, test=0):
        self.net = ppt.create_empty_network(fluid="air")

        if test == 1: self.path_prefix = '.'
        else: self.path_prefix = '..'

        '''
        false_some_sinks_test = 0
        if false_some_sinks_test == 1:
            for i in range(18):
                self.net.sink.at[i, 'in_service'] = False
        '''
        self.state_space_ids = ['sinks_m', 'mass_storage_mass_percent', 'gas_price']
        self.action_space_ids = ['CHP_m', 'Heat_Pump_m', 'Natural_Gas_Boiler_m', 'mass_storage_m']
        self.future_style = future_style
        self.compress_style = compress_style
        self.future_scale = future_scale
        self.time_step = 0
        if self.future_style == 'random_noise' and self.compress_style == 'PCA':
            self.compress_sinks_m_data = pd.read_csv(f'{self.path_prefix}/state_compress/randomized_{self.future_scale}_reduced_states/sink_m_randomized_{self.future_scale}_reduced_states.csv', usecols=['reduced_states'])
        elif self.future_style == 'LSTM_predict' and self.compress_style == 'PCA':
            self.compress_sinks_m_data = pd.read_csv(f'{self.path_prefix}/state_compress/LSTM_predict_{self.future_scale}_reduced_states/sink_m_predict_{self.future_scale}_reduced_states.csv', usecols=['reduced_states'])
        self.gas_price_data = pd.read_csv(f'{self.path_prefix}/data/profile/gas_price_profile.csv', usecols=['price']) # ?

        self.compress_sinks_m = self.compress_sinks_m_data.iloc[0, 0]
        self.compress_gas_price = self.gas_price_data.iloc[0, 0]

        self.sinks_m = pd.read_csv(f'{self.path_prefix}/data/profile/sink_m_profile.csv')
        self.gas_price = pd.read_csv(f'{self.path_prefix}/data/profile/gas_price_profile.csv', usecols=['price']) * 23.01 * 28.317

        # Create junctions
        junctions = ppt.create_junctions(self.net, 33, pn_bar=12, tfluid_k=303.15, name=[f'{i}' for i in range(0, 33)], type='j')

        # Create pipes, using same length and standard type
        # Main branch
        ppt.create_pipes(self.net, [junctions[i] for i in range(0, 17)], [junctions[i] for i in range(1, 18)], length_km=1.0, alpha_w_per_m2k=100, std_type="350_GGG", name=[f'line {i}-{i + 1}' for i in range(1, 18)])
        # Side branch 1
        ppt.create_pipes(self.net, [junctions[i] for i in range(18, 21)], [junctions[i] for i in range(19, 22)], length_km=1.0, alpha_w_per_m2k=100, std_type="350_GGG", name=[f'line {i}-{i + 1}' for i in range(19, 22)])
        # Side branch 2
        ppt.create_pipes(self.net, [junctions[i] for i in range(22, 24)], [junctions[i] for i in range(23, 25)], length_km=1.0, alpha_w_per_m2k=100, std_type="350_GGG", name=[f'line {i}-{i + 1}' for i in range(23, 25)])
        # Side branch 3
        ppt.create_pipes(self.net, [junctions[i] for i in range(25, 32)], [junctions[i] for i in range(26, 33)], length_km=1.0, alpha_w_per_m2k=100, std_type="350_GGG", name=[f'line {i}-{i + 1}' for i in range(26, 33)])
        # Connections between branches
        ppt.create_pipe(self.net, junctions[1], junctions[18], length_km=1.0, alpha_w_per_m2k=100, std_type="350_GGG", name='line 2-19')
        ppt.create_pipe(self.net, junctions[2], junctions[22], length_km=1.0, alpha_w_per_m2k=100, std_type="350_GGG", name='line 3-23')
        ppt.create_pipe(self.net, junctions[5], junctions[25], length_km=1.0, alpha_w_per_m2k=100, std_type="350_GGG", name='line 6-26')

        # Create external grid
        ppt.create_ext_grid(self.net, junctions[32], p_bar=14, t_k=303.15, name="Grid Connection")

        # Create mass storages
        self.Th_Bat1_E = Th_Bat1_E
        self.mass_storage = ppt.create_mass_storage(self.net, junctions[21], mdot_kg_per_s=0.0, max_m_stored_kg=Th_Bat1_E, init_m_stored_kg=0, name='Thermal Battery')

        # Create source
        self.CHP_thermal = ppt.create_source(self.net, junctions[0], mdot_kg_per_s=0.0, name="CHP")
        self.Heat_Pump = ppt.create_source(self.net, junctions[5], mdot_kg_per_s=0.0, name="Heat Pump")
        self.Natural_Gas_Boiler = ppt.create_source(self.net, junctions[24], mdot_kg_per_s=0.0, name="Natural Gas Boiler")

        # Create sinks
        self.sink1  = ppt.create_sink(self.net, junctions[1],  mdot_kg_per_s=0.0,  name='sink 1',  in_service=True)
        self.sink4  = ppt.create_sink(self.net, junctions[3],  mdot_kg_per_s=0.0,  name='sink 4',  in_service=True)
        self.sink7  = ppt.create_sink(self.net, junctions[6],  mdot_kg_per_s=0.0,  name='sink 7',  in_service=True)
        self.sink8  = ppt.create_sink(self.net, junctions[7],  mdot_kg_per_s=0.0,  name='sink 8',  in_service=True)
        self.sink10 = ppt.create_sink(self.net, junctions[9],  mdot_kg_per_s=0.0,  name='sink 10', in_service=True)
        self.sink11 = ppt.create_sink(self.net, junctions[10], mdot_kg_per_s=0.0,  name='sink 11', in_service=True)
        self.sink13 = ppt.create_sink(self.net, junctions[12], mdot_kg_per_s=0.0,  name='sink 13', in_service=True)
        self.sink14 = ppt.create_sink(self.net, junctions[13], mdot_kg_per_s=0.0,  name='sink 14', in_service=True)
        self.sink16 = ppt.create_sink(self.net, junctions[15], mdot_kg_per_s=0.0,  name='sink 16', in_service=True)
        self.sink17 = ppt.create_sink(self.net, junctions[16], mdot_kg_per_s=0.0,  name='sink 17', in_service=True)
        self.sink20 = ppt.create_sink(self.net, junctions[19], mdot_kg_per_s=0.0,  name='sink 20', in_service=True)
        self.sink21 = ppt.create_sink(self.net, junctions[20], mdot_kg_per_s=0.0,  name='sink 21', in_service=True)
        self.sink23 = ppt.create_sink(self.net, junctions[22], mdot_kg_per_s=0.0,  name='sink 23', in_service=True)
        self.sink24 = ppt.create_sink(self.net, junctions[23], mdot_kg_per_s=0.0,  name='sink 24', in_service=True)
        self.sink26 = ppt.create_sink(self.net, junctions[25], mdot_kg_per_s=0.0,  name='sink 26', in_service=True)
        self.sink27 = ppt.create_sink(self.net, junctions[26], mdot_kg_per_s=0.0,  name='sink 27', in_service=True)
        self.sink29 = ppt.create_sink(self.net, junctions[28], mdot_kg_per_s=0.0,  name='sink 29', in_service=True)
        self.sink30 = ppt.create_sink(self.net, junctions[29], mdot_kg_per_s=0.0,  name='sink 30', in_service=True)
        self.sink31 = ppt.create_sink(self.net, junctions[30], mdot_kg_per_s=0.0,  name='sink 31', in_service=True)
        self.sink32 = ppt.create_sink(self.net, junctions[31], mdot_kg_per_s=0.0,  name='sink 32', in_service=True)

    def modify_values(self, mass_storage_m, mass_storage_mass_percent, CHP_m, Heat_Pump_m, Natural_Gas_Boiler_m, sinks_m):
        self.net.mass_storage.at[0, 'mdot_kg_per_s'] = mass_storage_m
        #print('mass_storage_m: ', mass_storage_m)
        self.net.mass_storage.at[0, 'init_m_stored_kg'] = mass_storage_mass_percent
        #print('mass_storage_mass_percent: ', mass_storage_mass_percent)
        self.net.source.at[0, 'mdot_kg_per_s'] = CHP_m
        #print('CHP_m: ', CHP_m)
        self.net.source.at[1, 'mdot_kg_per_s'] = Heat_Pump_m
        #print('Heat_Pump_m: ', Heat_Pump_m)
        self.net.source.at[2, 'mdot_kg_per_s'] = Natural_Gas_Boiler_m
        #print('Natural_Gas_Boiler_m: ', Natural_Gas_Boiler_m)
        for i in range(20):
            self.net.sink.at[i, 'mdot_kg_per_s']  = sinks_m[i]
        #print('sinks_m: ', sinks_m)

    def run_flow(self):
        ppt.pipeflow(self.net)

    def get_next_remaining_mass(self):
        # calculating mass_storage_percent of battery at next step
        self.mass_storage_m = self.net.mass_storage['mdot_kg_per_s'][0]
        self.mass_storage_max_mass = self.net.mass_storage['max_m_stored_kg'][0]
        self.mass_storage_mass_percent = self.net.mass_storage['init_m_stored_kg'][0]
        #print('mass_storage_mass_percent: ', self.mass_storage_mass_percent)
        #print('mass_storage_m: ', self.mass_storage_m)
        #print('mass_storage_max_mass: ', self.mass_storage_max_mass)
        self.mass_storage_next_mass_percent = self.mass_storage_mass_percent + (self.mass_storage_m * 60 * 60) / self.mass_storage_max_mass * 100
        return self.mass_storage_next_mass_percent

    def simple_plot(self):
        ppt.plotting.simple_plot(self.net, plot_sinks=True, plot_sources=True, pump_size=0.5)

if __name__ == "__main__":
    """
    Functional Test
    """
    thermal_net = thermal_env_v2(future_style='random_noise', compress_style='PCA', future_scale=1, test=1)
    time_step = 0
    mass_storage_m = -1.0692246e-03
    mass_storage_mass_percent = 50
    CHP_m = 9.8490725e+00 * 1503 /60/60
    Heat_Pump_m = 5.3270044e+00
    Natural_Gas_Boiler_m = 3.0491948e-02
    sinks_m = thermal_net.sinks_m.loc[time_step,:].values[1:]
    thermal_net.modify_values(mass_storage_m, mass_storage_mass_percent, CHP_m, Heat_Pump_m, Natural_Gas_Boiler_m, sinks_m)
    thermal_net.run_flow()
    thermal_net.simple_plot()