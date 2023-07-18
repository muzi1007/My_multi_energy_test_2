import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

import pandapower as ppe
import pandapipes as ppth

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
    ppe.create_ext_grid(net, buses[0], vm_pu=1.02, va_degree=50)

    # Create batteries
    bat1 = ppe.create_storage(net, buses[32], dis_charge[0], max_e_mwh=Bat1_E, max_p_mw=Bat1_c_max, min_p_mw=Bat1_d_min, name='Bat 1')

    # Create PVs
    pv1 = ppe.create_sgen(net, buses[21], PV_profile[0], q_mvar=0, name='PV 1', type='PV')

    # Create Wind turbines
    Wind1 = ppe.create_sgen(net, buses[17], Wind_profile[0], q_mvar=0, name='Wind turbine 1', type='Wind turbine')

    # Create loads
    load1  = ppe.create_load(net, buses[0],  load_profile[0],  a_mvar=0, name='load 1',  in_service=True)
    load2  = ppe.create_load(net, buses[3],  load_profile[1],  a_mvar=0, name='load 2',  in_service=True)
    load3  = ppe.create_load(net, buses[4],  load_profile[2],  a_mvar=0, name='load 3',  in_service=True)
    load4  = ppe.create_load(net, buses[6],  load_profile[3],  a_mvar=0, name='load 4',  in_service=True)
    load5  = ppe.create_load(net, buses[8],  load_profile[4],  a_mvar=0, name='load 5',  in_service=True)
    load6  = ppe.create_load(net, buses[9],  load_profile[5],  a_mvar=0, name='load 6',  in_service=True)
    load7  = ppe.create_load(net, buses[10], load_profile[6],  a_mvar=0, name='load 7',  in_service=True)
    load8  = ppe.create_load(net, buses[11], load_profile[7],  a_mvar=0, name='load 8',  in_service=True)
    load9  = ppe.create_load(net, buses[13], load_profile[8],  a_mvar=0, name='load 9',  in_service=True)
    load10 = ppe.create_load(net, buses[14], load_profile[9],  a_mvar=0, name='load 10', in_service=True)
    load11 = ppe.create_load(net, buses[15], load_profile[10], a_mvar=0, name='load 11', in_service=True)
    load12 = ppe.create_load(net, buses[17], load_profile[11], a_mvar=0, name='load 12', in_service=True)
    load13 = ppe.create_load(net, buses[18], load_profile[12], a_mvar=0, name='load 13', in_service=True)
    load14 = ppe.create_load(net, buses[21], load_profile[13], a_mvar=0, name='load 14', in_service=True)
    load15 = ppe.create_load(net, buses[23], load_profile[14], a_mvar=0, name='load 15', in_service=True)
    load16 = ppe.create_load(net, buses[25], load_profile[15], a_mvar=0, name='load 16', in_service=True)
    load17 = ppe.create_load(net, buses[27], load_profile[16], a_mvar=0, name='load 17', in_service=True)
    load18 = ppe.create_load(net, buses[29], load_profile[17], a_mvar=0, name='load 18', in_service=True)
    load19 = ppe.create_load(net, buses[30], load_profile[18], a_mvar=0, name='load 19', in_service=True)
    load20 = ppe.create_load(net, buses[31], load_profile[19], a_mvar=0, name='load 20', in_service=True)

    ids = {
        'pv1': pv1,
        'Wind1': Wind1,
        'bat1': bat1,
        'load1': load1, 'load2': load2, 'load3': load3, 'load4': load4, 'load5': load5, 'load6': load6, 'load7': load7, 'load8': load8,
        'load9': load9, 'load10': load10, 'load11': load11, 'load12': load12, 'load13': load13, 'load14': load14, 'load15': load15,
        'load16': load16, 'load17': load17, 'load18': load18, 'load19': load19, 'load20': load20
    }

    return net, ids

# Create thermal grid environment
def create_thermal_env(sink_profile):
    # Create thermal net
    net = ppth.create_empty_network()

    # Create junctions
    junctions = ppth.create_junctions(net, 33, pn_bar=5, tfluid_k=320, name=[f'Bus {i}' for i in range(1, 33 + 1)], type='j')

    # Create pipes, using same length and standard type
    # Main branch
    ppth.create_pipes(net, [junctions[i] for i in range(0, 17)], [junctions[i] for i in range(1, 18)], length_km=1.0,
                     std_type="80_GGG", name=[f'line {i}-{i + 1}' for i in range(1, 18)])
    # Side branch 1
    ppth.create_pipes(net, [junctions[i] for i in range(18, 21)], [junctions[i] for i in range(19, 22)], length_km=1.0,
                     std_type="80_GGG", name=[f'line {i}-{i + 1}' for i in range(19, 22)])
    # Side branch 2
    ppth.create_pipes(net, [junctions[i] for i in range(22, 24)], [junctions[i] for i in range(23, 25)], length_km=1.0,
                     std_type="80_GGG", name=[f'line {i}-{i + 1}' for i in range(23, 25)])
    # Side branch 3
    ppth.create_pipes(net, [junctions[i] for i in range(25, 32)], [junctions[i] for i in range(26, 33)], length_km=1.0,
                     std_type="80_GGG", name=[f'line {i}-{i + 1}' for i in range(26, 33)])

    # Connections between branches
    ppth.create_pipe(net, junctions[1], junctions[18], length_km=1.0, std_type="80_GGG", name='line 2-19')
    ppth.create_pipe(net, junctions[2], junctions[22], length_km=1.0, std_type="80_GGG", name='line 3-23')
    ppth.create_pipe(net, junctions[5], junctions[25], length_km=1.0, std_type="80_GGG", name='line 6-26')

    temp_junction = ppth.create_junction(net, pn_bar=5, tfluid_k=320, name='temp_junction', type='j')
    ppth.create_pipe(net, temp_junction, junctions[24], length_km=1.0, std_type="80_GGG", name='line 6-26')
    ppth.create_ext_grid(net, temp_junction, p_bar=100, t_k=340)
    ppth.create_heat_exchanger(net, temp_junction, junctions[24], diameter_m=40e-3, qext_w=2000) # qext_w ?

    # Create sinks
    sink1  = ppth.create_sink(net, junctions[0],  sink_profile[0],  a_mvar=0, name='sink 1',  in_service=True)
    sink2  = ppth.create_sink(net, junctions[3],  sink_profile[1],  a_mvar=0, name='sink 2',  in_service=True)
    sink3  = ppth.create_sink(net, junctions[4],  sink_profile[2],  a_mvar=0, name='sink 3',  in_service=True)
    sink4  = ppth.create_sink(net, junctions[6],  sink_profile[3],  a_mvar=0, name='sink 4',  in_service=True)
    sink5  = ppth.create_sink(net, junctions[8],  sink_profile[4],  a_mvar=0, name='sink 5',  in_service=True)
    sink6  = ppth.create_sink(net, junctions[9],  sink_profile[5],  a_mvar=0, name='sink 6',  in_service=True)
    sink7  = ppth.create_sink(net, junctions[10], sink_profile[6],  a_mvar=0, name='sink 7',  in_service=True)
    sink8  = ppth.create_sink(net, junctions[11], sink_profile[7],  a_mvar=0, name='sink 8',  in_service=True)
    sink9  = ppth.create_sink(net, junctions[13], sink_profile[8],  a_mvar=0, name='sink 9',  in_service=True)
    sink10 = ppth.create_sink(net, junctions[14], sink_profile[9],  a_mvar=0, name='sink 10', in_service=True)
    sink11 = ppth.create_sink(net, junctions[15], sink_profile[10], a_mvar=0, name='sink 11', in_service=True)
    sink12 = ppth.create_sink(net, junctions[17], sink_profile[11], a_mvar=0, name='sink 12', in_service=True)
    sink13 = ppth.create_sink(net, junctions[18], sink_profile[12], a_mvar=0, name='sink 13', in_service=True)
    sink14 = ppth.create_sink(net, junctions[21], sink_profile[13], a_mvar=0, name='sink 14', in_service=True)
    sink15 = ppth.create_sink(net, junctions[23], sink_profile[14], a_mvar=0, name='sink 15', in_service=True)
    sink16 = ppth.create_sink(net, junctions[25], sink_profile[15], a_mvar=0, name='sink 16', in_service=True)
    sink17 = ppth.create_sink(net, junctions[27], sink_profile[16], a_mvar=0, name='sink 17', in_service=True)
    sink18 = ppth.create_sink(net, junctions[29], sink_profile[17], a_mvar=0, name='sink 18', in_service=True)
    sink19 = ppth.create_sink(net, junctions[30], sink_profile[18], a_mvar=0, name='sink 19', in_service=True)
    sink20 = ppth.create_sink(net, junctions[31], sink_profile[19], a_mvar=0, name='sink 20', in_service=True)

    ids = {
        'sink1': sink1, 'sink2': sink2, 'sink3': sink3, 'sink4': sink4, 'sink5': sink5, 'sink6': sink6, 'sink7': sink7, 'sink8': sink8,
        'sink9': sink9, 'sink10': sink10, 'sink11': sink11, 'sink12': sink12, 'sink13': sink13, 'sink14': sink14, 'sink15': sink15,
        'sink16': sink16, 'sink17': sink17, 'sink18': sink18, 'sink19': sink19, 'sink20': sink20
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
        from parameters import *

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

    ppe.plotting.simple_plot(electric_net, plot_sgens=True, plot_loads=True)
    ppe.plotting.to_html(electric_net, 'env/electric_net_test.html')


    thermal_net, ids = create_thermal_env(sink_max_p_list)
    ppth.plotting.simple_plot(thermal_net, plot_sinks=True)