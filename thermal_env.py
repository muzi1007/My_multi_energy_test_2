import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

import pandapower as ppe
import pandapipes as ppth

from parameters import *

# Create thermal grid environment
def create_thermal_env(sink_profile, dis_charge, CHP_thermal_input):
    # Create thermal net
    net = ppth.create_empty_network(fluid="air")

    # Create junctions
    junctions = ppth.create_junctions(net, 33, pn_bar=1, tfluid_k=293.15, name=[f'Bus {i}' for i in range(1, 33 + 1)], type='j')

    # Create pipes, using same length and standard type
    # Main branch
    ppth.create_pipes(net, [junctions[i] for i in range(0, 17)], [junctions[i] for i in range(1, 18)], length_km=1.0, alpha_w_per_m2k=100,
                     std_type="80_GGG", name=[f'line {i}-{i + 1}' for i in range(1, 18)])
    # Side branch 1
    ppth.create_pipes(net, [junctions[i] for i in range(18, 21)], [junctions[i] for i in range(19, 22)], length_km=1.0, alpha_w_per_m2k=100,
                     std_type="80_GGG", name=[f'line {i}-{i + 1}' for i in range(19, 22)])
    # Side branch 2
    ppth.create_pipes(net, [junctions[i] for i in range(22, 24)], [junctions[i] for i in range(23, 25)], length_km=1.0, alpha_w_per_m2k=100,
                     std_type="80_GGG", name=[f'line {i}-{i + 1}' for i in range(23, 25)])
    # Side branch 3
    ppth.create_pipes(net, [junctions[i] for i in range(25, 32)], [junctions[i] for i in range(26, 33)], length_km=1.0, alpha_w_per_m2k=100,
                     std_type="80_GGG", name=[f'line {i}-{i + 1}' for i in range(26, 33)])

    # Connections between branches
    ppth.create_pipe(net, junctions[1], junctions[18], length_km=1.0, alpha_w_per_m2k=100,  std_type="80_GGG", name='line 2-19')
    ppth.create_pipe(net, junctions[2], junctions[22], length_km=1.0, alpha_w_per_m2k=100,  std_type="80_GGG", name='line 3-23')
    ppth.create_pipe(net, junctions[5], junctions[25], length_km=1.0, alpha_w_per_m2k=100,  std_type="80_GGG", name='line 6-26')

    # Create pumps
    #ppth.create_circ_pump_const_pressure(net, junctions[21], junctions[20], p_flow_bar=10, plift_bar=2, t_flow_k=293.15)
    #ppth.create_circ_pump_const_pressure(net, junctions[5], junctions[4], p_flow_bar=10, plift_bar=2, t_flow_k=293.15)

    # Create source
    ppth.create_source(net, junctions[0], mdot_kg_per_s=CHP_thermal_input, name="CHP")

    # Create external grid
    ppth.create_ext_grid(net, junctions[24], p_bar=10, t_k=293.15, name="Grid Connection")
    ppth.create_ext_grid(net, junctions[5], p_bar=10, t_k=293.15, name="Heat Pump")

    # Create mass storages
    mass_storage1 = ppth.create_mass_storage(net, junctions[21], dis_charge[0], max_m_stored_kg=th_Bat1_E, name='Thermal Battery')

    # Create sinks
    sink1  = ppth.create_sink(net, junctions[0],  sink_profile[0],  name='sink 1',  in_service=True)
    sink2  = ppth.create_sink(net, junctions[3],  sink_profile[1],  name='sink 2',  in_service=True)
    sink3  = ppth.create_sink(net, junctions[5],  sink_profile[2],  name='sink 3',  in_service=True)
    sink4  = ppth.create_sink(net, junctions[6],  sink_profile[3],  name='sink 4',  in_service=True)
    sink5  = ppth.create_sink(net, junctions[7],  sink_profile[4],  name='sink 5',  in_service=True)
    sink6  = ppth.create_sink(net, junctions[9],  sink_profile[5],  name='sink 6',  in_service=True)
    sink7  = ppth.create_sink(net, junctions[10], sink_profile[6],  name='sink 7',  in_service=True)
    sink8  = ppth.create_sink(net, junctions[12], sink_profile[7],  name='sink 8',  in_service=True)
    sink9  = ppth.create_sink(net, junctions[13], sink_profile[8],  name='sink 9',  in_service=True)
    sink10 = ppth.create_sink(net, junctions[15], sink_profile[9],  name='sink 10', in_service=True)

    ids = {
        'mass_storage1': mass_storage1,
        'sink1': sink1, 'sink2': sink2, 'sink3': sink3, 'sink4': sink4, 'sink5': sink5, 'sink6': sink6, 'sink7': sink7, 'sink8': sink8,
        'sink9': sink9, 'sink10': sink10
    }

    return net, ids

if __name__ == "__main__":
    """
    Functional Test
    """
    thermal_net, ids = create_thermal_env(sink_max_p_list, th_Bat_d_min_list, 3.133203)
    ppth.plotting.simple_plot(thermal_net, plot_sinks=True, plot_sources=True, pump_size=0.5)

    ppth.pipeflow(thermal_net)