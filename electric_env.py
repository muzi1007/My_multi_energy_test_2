import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

import pandapower as ppe
import pandapipes as ppth
from pandapower.control import ConstControl
from pandapower.timeseries.data_sources.frame_data import DFData
import pandapower.timeseries as ts
import pandapower.timeseries.output_writer as ow

from parameters import *

# Create electric grid environment
def create_electric_env(PV_ds, Wind_ds, load_ds, load_state = None):
    # Create electric net
    net = ppe.create_empty_network()

    # Create buses
    buses = ppe.create_buses(net, 33, vn_kv=110, name=[f'Bus {i}' for i in range(1, 33 + 1)], type='b')

    # Create lines, using same length and standard type
    # Main branch
    ppe.create_lines(net, [buses[i] for i in range(0, 17)], [buses[i] for i in range(1, 18)], length_km=1.0,
                    std_type="NAYY 4x120 SE", name=[f'line {i}-{i + 1}' for i in range(1, 18)])
    # Side branch 1
    ppe.create_lines(net, [buses[i] for i in range(18, 21)], [buses[i] for i in range(19, 22)], length_km=1.0,
                    std_type="NAYY 4x120 SE", name=[f'line {i}-{i + 1}' for i in range(19, 22)])
    # Side branch 2
    ppe.create_lines(net, [buses[i] for i in range(22, 24)], [buses[i] for i in range(23, 25)], length_km=1.0,
                    std_type="NAYY 4x120 SE", name=[f'line {i}-{i + 1}' for i in range(23, 25)])
    # Side branch 3
    ppe.create_lines(net, [buses[i] for i in range(25, 32)], [buses[i] for i in range(26, 33)], length_km=1.0,
                    std_type="NAYY 4x120 SE", name=[f'line {i}-{i + 1}' for i in range(26, 33)])

    # Connections between branches
    ppe.create_line(net, buses[1], buses[18], length_km=1.0, std_type="NAYY 4x120 SE", name='line 2-19')
    ppe.create_line(net, buses[2], buses[22], length_km=1.0, std_type="NAYY 4x120 SE", name='line 3-23')
    ppe.create_line(net, buses[5], buses[25], length_km=1.0, std_type="NAYY 4x120 SE", name='line 6-26')

    # Create external grid
    ppe.create_ext_grid(net, buses[0], name='Grid Connection')
    ppe.create_ext_grid(net, buses[0], name='CHP') # 要改sgen?

    # Create batteries
    battery1 = ppe.create_storage(net, buses[32], 0.0, max_e_mwh=E_Bat1_E, max_p_mw=E_Bat1_c_max, min_p_mw=E_Bat1_d_min, name='Electric Battery')

    # Create PVs
    pv1 = ppe.create_sgen(net, buses[21], 0.0, q_mvar=0, name='PV 1', type='PV')
    ConstControl(net, element='sgen', variable='p_mw', element_index=pv1, profile_name='electricity', data_source=PV_ds)

    # Create Wind turbines
    wind1 = ppe.create_sgen(net, buses[17], 0.0, q_mvar=0, name='Wind turbine 1', type='Wind Turbine')
    ConstControl(net, element='sgen', variable='p_mw', element_index=wind1, profile_name='electricity', data_source=Wind_ds)

    # Create loads
    load1  =  ppe.create_load(net, buses[0],  0.0, q_mvar=0, name='load 1',  in_service=True)
    load4  =  ppe.create_load(net, buses[3],  0.0, q_mvar=0, name='load 4',  in_service=True)
    load5  =  ppe.create_load(net, buses[4],  0.0, q_mvar=0, name='load 5',  in_service=True)
    load7  =  ppe.create_load(net, buses[6],  0.0, q_mvar=0, name='load 7',  in_service=True)
    load9  =  ppe.create_load(net, buses[8],  0.0, q_mvar=0, name='load 9',  in_service=True)
    load10 =  ppe.create_load(net, buses[9],  0.0, q_mvar=0, name='load 10', in_service=True)
    load11 =  ppe.create_load(net, buses[10], 0.0, q_mvar=0, name='load 11', in_service=True)
    load12 =  ppe.create_load(net, buses[11], 0.0, q_mvar=0, name='load 12', in_service=True)
    load14 =  ppe.create_load(net, buses[13], 0.0, q_mvar=0, name='load 14', in_service=True)
    load15 =  ppe.create_load(net, buses[14], 0.0, q_mvar=0, name='load 15', in_service=True)
    load16 =  ppe.create_load(net, buses[15], 0.0, q_mvar=0, name='load 16', in_service=True)
    load18 =  ppe.create_load(net, buses[17], 0.0, q_mvar=0, name='load 18', in_service=True)
    load19 =  ppe.create_load(net, buses[18], 0.0, q_mvar=0, name='load 19', in_service=True)
    load23 =  ppe.create_load(net, buses[21], 0.0, q_mvar=0, name='load 23', in_service=True)
    load24 =  ppe.create_load(net, buses[23], 0.0, q_mvar=0, name='load 24', in_service=True)
    load26 =  ppe.create_load(net, buses[25], 0.0, q_mvar=0, name='load 26', in_service=True)
    load28 =  ppe.create_load(net, buses[27], 0.0, q_mvar=0, name='load 28', in_service=True)
    load30 =  ppe.create_load(net, buses[29], 0.0, q_mvar=0, name='load 30', in_service=True)
    load31 =  ppe.create_load(net, buses[30], 0.0, q_mvar=0, name='load 31', in_service=True)
    load32 =  ppe.create_load(net, buses[31], 0.0, q_mvar=0, name='load 32', in_service=True)
    ConstControl(net, element='load', variable='p_mw', element_index=load1,  profile_name='load1',  data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load4,  profile_name='load4',  data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load5,  profile_name='load5',  data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load7,  profile_name='load7',  data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load9,  profile_name='load9',  data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load10, profile_name='load10', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load11, profile_name='load11', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load12, profile_name='load12', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load14, profile_name='load14', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load15, profile_name='load15', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load16, profile_name='load16', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load18, profile_name='load18', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load19, profile_name='load19', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load23, profile_name='load23', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load24, profile_name='load24', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load26, profile_name='load26', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load28, profile_name='load28', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load30, profile_name='load30', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load31, profile_name='load31', data_source=load_ds)
    ConstControl(net, element='load', variable='p_mw', element_index=load32, profile_name='load32', data_source=load_ds)

    ids = {
        'pv1': pv1,
        'wind1': wind1,
        'battery1': battery1,
        'load1': load1, 'load4': load4, 'load5': load5, 'load7': load7, 'load9': load9, 'load10': load10, 'load11': load11, 'load12': load12,
        'load14': load14, 'load15': load15, 'load16': load16, 'load18': load18, 'load19': load19, 'load23': load23, 'load24': load24,
        'load26': load26, 'load28': load28, 'load30': load30, 'load31': load31, 'load32': load32
    }

    return net, ids

if __name__ == "__main__":
    """
    Functional Test
    """
    simulation_hours = 72
    PV_profile = pd.read_csv('./data/profile/PV_profile.csv')
    Wind_profile = pd.read_csv('./data/profile/Wind_profile.csv')
    load_profile = pd.read_csv('./data/profile/load_profile.csv')
    PV_ds = DFData(PV_profile.iloc[0: simulation_hours])
    Wind_ds = DFData(Wind_profile.iloc[0: simulation_hours])
    load_ds = DFData(load_profile.iloc[0: simulation_hours])

    #print('hour ', i)
    electric_net, ids = create_electric_env(PV_ds, Wind_ds, load_ds)
    owr = ow.OutputWriter(electric_net, output_path="./data/3_days_test_without_actions/output_writer/electric_net", output_file_type=".csv", csv_separator=',')
    owr.log_variable('res_ext_grid', 'p_mw')
    owr.log_variable('res_load', 'p_mw')
    ts.run_timeseries(electric_net, time_steps=simulation_hours, continue_on_divergence=False)


    #ppe.plotting.simple_plot(electric_net, plot_gens=True, plot_sgens=True, plot_line_switches=True, plot_loads=True)
    #ppe.plotting.to_html(electric_net, 'env/electric_net_test.html')