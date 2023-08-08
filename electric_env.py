import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

import pandapower as ppe
import pandapipes as ppth

from parameters import *

# Create electric grid environment
def create_electric_env(PV_profile, Wind_profile, load_profile, dis_charge, load_state = None):
    # Create electric net
    net = ppe.create_empty_network()

    # Create buses
    buses = ppe.create_buses(net, 33, vn_kv=110, name=[f'Bus {i}' for i in range(1, 33 + 1)], type='b')

    # Create lines, using same length and standard type
    # Main branch
    ppe.create_lines(net, [buses[i] for i in range(0, 17)], [buses[i] for i in range(1, 18)], length_km=1.0,
                    std_type="NAYY 4x50 SE", name=[f'line {i}-{i + 1}' for i in range(1, 18)])
    # Side branch 1
    ppe.create_lines(net, [buses[i] for i in range(18, 21)], [buses[i] for i in range(19, 22)], length_km=1.0,
                    std_type="NAYY 4x50 SE", name=[f'line {i}-{i + 1}' for i in range(19, 22)])
    # Side branch 2
    ppe.create_lines(net, [buses[i] for i in range(22, 24)], [buses[i] for i in range(23, 25)], length_km=1.0,
                    std_type="NAYY 4x50 SE", name=[f'line {i}-{i + 1}' for i in range(23, 25)])
    # Side branch 3
    ppe.create_lines(net, [buses[i] for i in range(25, 32)], [buses[i] for i in range(26, 33)], length_km=1.0,
                    std_type="NAYY 4x50 SE", name=[f'line {i}-{i + 1}' for i in range(26, 33)])

    # Connections between branches
    ppe.create_line(net, buses[1], buses[18], length_km=1.0, std_type="NAYY 4x50 SE", name='line 2-19')
    ppe.create_line(net, buses[2], buses[22], length_km=1.0, std_type="NAYY 4x50 SE", name='line 3-23')
    ppe.create_line(net, buses[5], buses[25], length_km=1.0, std_type="NAYY 4x50 SE", name='line 6-26')

    # Create external grid
    ppe.create_ext_grid(net, buses[0], vm_pu=1.02, va_degree=50, name='Grid Connection')
    ppe.create_ext_grid(net, buses[0], vm_pu=1.02, va_degree=50, name='CHP')

    # Create batteries
    battery1 = ppe.create_storage(net, buses[32], dis_charge[0], max_e_mwh=Bat1_E, max_p_mw=Bat1_c_max, min_p_mw=Bat1_d_min, name='Electric Battery')

    # Create PVs
    pv1 = ppe.create_sgen(net, buses[21], PV_profile[0], q_mvar=0, name='PV 1', type='PV')

    # Create Wind turbines
    Wind1 = ppe.create_sgen(net, buses[17], Wind_profile[0], q_mvar=0, name='Wind turbine 1', type='Wind Turbine')

    # Create loads
    load1  =  ppe.create_load(net, buses[0],  load_profile[0],  a_mvar=0, name='load 1',  in_service=True)
    load4  =  ppe.create_load(net, buses[3],  load_profile[1],  a_mvar=0, name='load 4',  in_service=True)
    load5  =  ppe.create_load(net, buses[4],  load_profile[2],  a_mvar=0, name='load 5',  in_service=True)
    load7  =  ppe.create_load(net, buses[6],  load_profile[3],  a_mvar=0, name='load 7',  in_service=True)
    load9  =  ppe.create_load(net, buses[8],  load_profile[4],  a_mvar=0, name='load 9',  in_service=True)
    load10 =  ppe.create_load(net, buses[9],  load_profile[5],  a_mvar=0, name='load 10', in_service=True)
    load11 =  ppe.create_load(net, buses[10], load_profile[6],  a_mvar=0, name='load 11', in_service=True)
    load12 =  ppe.create_load(net, buses[11], load_profile[7],  a_mvar=0, name='load 12', in_service=True)
    load14 =  ppe.create_load(net, buses[13], load_profile[8],  a_mvar=0, name='load 14', in_service=True)
    load15 =  ppe.create_load(net, buses[14], load_profile[9],  a_mvar=0, name='load 15', in_service=True)
    load16 =  ppe.create_load(net, buses[15], load_profile[10], a_mvar=0, name='load 16', in_service=True)
    load18 =  ppe.create_load(net, buses[17], load_profile[11], a_mvar=0, name='load 18', in_service=True)
    load19 =  ppe.create_load(net, buses[18], load_profile[12], a_mvar=0, name='load 19', in_service=True)
    load23 =  ppe.create_load(net, buses[21], load_profile[13], a_mvar=0, name='load 23', in_service=True)
    load24 =  ppe.create_load(net, buses[23], load_profile[14], a_mvar=0, name='load 24', in_service=True)
    load26 =  ppe.create_load(net, buses[25], load_profile[15], a_mvar=0, name='load 26', in_service=True)
    load28 =  ppe.create_load(net, buses[27], load_profile[16], a_mvar=0, name='load 28', in_service=True)
    load30 =  ppe.create_load(net, buses[29], load_profile[17], a_mvar=0, name='load 30', in_service=True)
    load31 =  ppe.create_load(net, buses[30], load_profile[18], a_mvar=0, name='load 31', in_service=True)
    load32 =  ppe.create_load(net, buses[31], load_profile[19], a_mvar=0, name='load 32', in_service=True)

    ids = {
        'pv1': pv1,
        'Wind1': Wind1,
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
    PV_units = 3300
    Wind_units = 6
    df_PV = pd.read_csv('env/renewable_energy_source/ninja_pv_23.7989_120.8553_ALL.csv', header=3)
    df_Wind = pd.read_csv('env/renewable_energy_source/ninja_wind_24.7373_120.8835_ALL.csv', header=3)

    res_ext_grid_record = []
    for i in range(0, simulation_hours):
        #print('hour ', i)

        PV1_max_p = df_PV.loc[i, 'electricity'] * 0.001 * PV_units
        pv_max_p_list = [PV1_max_p]

        Wind1_max_p = df_Wind.loc[i, 'electricity'] * 0.001 * Wind_units
        Wind_max_p_list = [Wind1_max_p]

        electric_net, ids = create_electric_env(pv_max_p_list, Wind_max_p_list, load_max_p_list, Bat_c_max_list)
        ppe.runpp(electric_net, numba=False)
        #print(net.res_ext_grid)
        res_ext_grid_record.append(electric_net.res_ext_grid.p_mw.values[0])


    plt.figure()
    plt.plot(range(0, simulation_hours), res_ext_grid_record)
    plt.xlabel('hours (hr)', size=12)
    plt.ylabel('Electricity (MW)', size=12)
    plt.legend(['res_ext_grid'], loc='upper right')
    plt.title('Electricity from external grid')

    ppe.plotting.simple_plot(electric_net, plot_gens=True, plot_sgens=True, plot_line_switches=True, plot_loads=True)
    #ppe.plotting.to_html(electric_net, 'env/electric_net_test.html')