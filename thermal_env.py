import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

import pandapower as ppe
import pandapipes as ppt

from parameters import *

# Create thermal grid environment
def create_thermal_env(sink_profile, dis_charge, CHP_thermal_input, simulation_hours):
    # Create thermal net
    net = ppt.create_empty_network(fluid="air")

    # Create junctions
    junctions = ppt.create_junctions(net, 33, pn_bar=12, tfluid_k=303.15, name=[f'Bus {i}' for i in range(1, 33 + 1)], type='j')

    # Create pipes, using same length and standard type
    # Main branch
    ppt.create_pipes(net, [junctions[i] for i in range(0, 17)], [junctions[i] for i in range(1, 18)], length_km=1.0, alpha_w_per_m2k=100,
                     std_type="80_GGG", name=[f'line {i}-{i + 1}' for i in range(1, 18)])
    # Side branch 1
    ppt.create_pipes(net, [junctions[i] for i in range(18, 21)], [junctions[i] for i in range(19, 22)], length_km=1.0, alpha_w_per_m2k=100,
                     std_type="80_GGG", name=[f'line {i}-{i + 1}' for i in range(19, 22)])
    # Side branch 2
    ppt.create_pipes(net, [junctions[i] for i in range(22, 24)], [junctions[i] for i in range(23, 25)], length_km=1.0, alpha_w_per_m2k=100,
                     std_type="80_GGG", name=[f'line {i}-{i + 1}' for i in range(23, 25)])
    # Side branch 3
    ppt.create_pipes(net, [junctions[i] for i in range(25, 32)], [junctions[i] for i in range(26, 33)], length_km=1.0, alpha_w_per_m2k=100,
                     std_type="80_GGG", name=[f'line {i}-{i + 1}' for i in range(26, 33)])

    # Connections between branches
    ppt.create_pipe(net, junctions[1], junctions[18], length_km=1.0, alpha_w_per_m2k=100,  std_type="80_GGG", name='line 2-19')
    ppt.create_pipe(net, junctions[2], junctions[22], length_km=1.0, alpha_w_per_m2k=100,  std_type="80_GGG", name='line 3-23')
    ppt.create_pipe(net, junctions[5], junctions[25], length_km=1.0, alpha_w_per_m2k=100,  std_type="80_GGG", name='line 6-26')

    # Create pumps
    #ppt.create_circ_pump_const_pressure(net, junctions[21], junctions[20], p_flow_bar=10, plift_bar=2, t_flow_k=293.15)
    #ppt.create_circ_pump_const_pressure(net, junctions[5], junctions[4], p_flow_bar=10, plift_bar=2, t_flow_k=293.15)

    # Create source
    ppt.create_source(net, junctions[0], mdot_kg_per_s=CHP_thermal_input, name="CHP")

    # Create external grid
    ppt.create_ext_grid(net, junctions[24], p_bar=14, t_k=303.15, name="Grid Connection")
    ppt.create_ext_grid(net, junctions[5], p_bar=14, t_k=303.15, name="Heat Pump") # 要改source?

    # Create mass storages
    mass_storage1 = ppt.create_mass_storage(net, junctions[21], dis_charge[0], max_m_stored_kg=Th_Bat1_E, name='Thermal Battery')

    # Create sinks
    sink1  = ppt.create_sink(net, junctions[0],  sink_profile['sink_m1'][simulation_hours],  name='sink 1',  in_service=True)
    sink4  = ppt.create_sink(net, junctions[3],  sink_profile['sink_m4'][simulation_hours],  name='sink 4',  in_service=True)
    sink7  = ppt.create_sink(net, junctions[6],  sink_profile['sink_m7'][simulation_hours],  name='sink 7',  in_service=True)
    sink8  = ppt.create_sink(net, junctions[7],  sink_profile['sink_m8'][simulation_hours],  name='sink 8',  in_service=True)
    sink10 = ppt.create_sink(net, junctions[9],  sink_profile['sink_m10'][simulation_hours],  name='sink 10', in_service=True)
    sink11 = ppt.create_sink(net, junctions[10], sink_profile['sink_m11'][simulation_hours],  name='sink 11', in_service=True)
    sink13 = ppt.create_sink(net, junctions[12], sink_profile['sink_m13'][simulation_hours],  name='sink 13', in_service=True)
    sink14 = ppt.create_sink(net, junctions[13], sink_profile['sink_m14'][simulation_hours],  name='sink 14', in_service=True)
    sink16 = ppt.create_sink(net, junctions[15], sink_profile['sink_m16'][simulation_hours],  name='sink 16', in_service=True)
    sink17 = ppt.create_sink(net, junctions[16], sink_profile['sink_m17'][simulation_hours],  name='sink 17', in_service=True)
    sink20 = ppt.create_sink(net, junctions[19], sink_profile['sink_m20'][simulation_hours],  name='sink 20', in_service=True)
    sink21 = ppt.create_sink(net, junctions[20], sink_profile['sink_m21'][simulation_hours],  name='sink 21', in_service=True)
    sink23 = ppt.create_sink(net, junctions[22], sink_profile['sink_m23'][simulation_hours],  name='sink 23', in_service=True)
    sink24 = ppt.create_sink(net, junctions[23], sink_profile['sink_m24'][simulation_hours],  name='sink 24', in_service=True)
    sink26 = ppt.create_sink(net, junctions[25], sink_profile['sink_m26'][simulation_hours],  name='sink 26', in_service=True)
    sink27 = ppt.create_sink(net, junctions[26], sink_profile['sink_m27'][simulation_hours],  name='sink 27', in_service=True)
    sink29 = ppt.create_sink(net, junctions[28], sink_profile['sink_m29'][simulation_hours],  name='sink 29', in_service=True)
    sink30 = ppt.create_sink(net, junctions[29], sink_profile['sink_m30'][simulation_hours],  name='sink 30', in_service=True)
    sink31 = ppt.create_sink(net, junctions[30], sink_profile['sink_m31'][simulation_hours],  name='sink 31', in_service=True)
    sink32 = ppt.create_sink(net, junctions[31], sink_profile['sink_m32'][simulation_hours],  name='sink 32', in_service=True)

    ids = {
        'mass_storage1': mass_storage1,
        'sink1': sink1, 'sink4': sink4, 'sink7': sink7, 'sink8': sink8, 'sink10': sink10, 'sink11': sink11, 'sink13': sink13, 'sink14': sink14, 'sink16': sink16, 'sink17': sink17,
        'sink20': sink20, 'sink21': sink21, 'sink23': sink23, 'sink24': sink24, 'sink26': sink26, 'sink27': sink27, 'sink29': sink29, 'sink30': sink30, 'sink31': sink31, 'sink32': sink32
    }

    return net, ids

if __name__ == "__main__":
    """
    Functional Test
    """
    simulation_hours = 72
    sink_m_profile = pd.read_csv('./data/profile/sink_m_profile.csv')

    res_ext_grid_record = []
    for i in range(0, simulation_hours):
        #print('hour ', i)
        thermal_net, ids = create_thermal_env(sink_m_profile, Th_Bat_d_min_list, 0, i) # CHP_thermal_input 未與 electric_net 整合
        ppt.pipeflow(thermal_net)
        res_ext_grid_record.append(thermal_net.res_ext_grid.mdot_kg_per_s.values[0])

    plt.figure()
    plt.plot(range(0, simulation_hours), res_ext_grid_record)
    plt.xlabel('hours (hr)', size=12)
    plt.ylabel('Mass (kg/s)', size=12)
    plt.legend(['res_ext_grid'], loc='upper right')
    plt.title('Mass from external grid')

    ppt.plotting.simple_plot(thermal_net, plot_sinks=True, plot_sources=True, pump_size=0.5)
