import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

import pandapower as ppe
import pandapipes as ppt
from pandapower.control import ConstControl
from pandapower.timeseries.data_sources.frame_data import DFData
import pandapipes.timeseries as ts
import pandapower.timeseries.output_writer as ow

from parameters import *

# Create thermal grid environment
def create_thermal_env(sink_ds, CHP_thermal_input_ds):
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
    CHP_thermal = ppt.create_source(net, junctions[0], 0.0, name="CHP")
    ConstControl(net, element='source', variable='mdot_kg_per_s', element_index=CHP_thermal, profile_name='1', data_source=CHP_thermal_input_ds)

    # Create external grid
    ppt.create_ext_grid(net, junctions[24], p_bar=14, t_k=303.15, name="Grid Connection")
    ppt.create_ext_grid(net, junctions[5], p_bar=14, t_k=303.15, name="Heat Pump") # 要改source?

    # Create mass storages
    mass_storage1 = ppt.create_mass_storage(net, junctions[21], 0.0, max_m_stored_kg=Th_Bat1_E, name='Thermal Battery')


    # Create sinks
    sink1  = ppt.create_sink(net, junctions[0],  0.0,  name='sink 1',  in_service=True)
    sink4  = ppt.create_sink(net, junctions[3],  0.0,  name='sink 4',  in_service=True)
    sink7  = ppt.create_sink(net, junctions[6],  0.0,  name='sink 7',  in_service=True)
    sink8  = ppt.create_sink(net, junctions[7],  0.0,  name='sink 8',  in_service=True)
    sink10 = ppt.create_sink(net, junctions[9],  0.0,  name='sink 10', in_service=True)
    sink11 = ppt.create_sink(net, junctions[10], 0.0,  name='sink 11', in_service=True)
    sink13 = ppt.create_sink(net, junctions[12], 0.0,  name='sink 13', in_service=True)
    sink14 = ppt.create_sink(net, junctions[13], 0.0,  name='sink 14', in_service=True)
    sink16 = ppt.create_sink(net, junctions[15], 0.0,  name='sink 16', in_service=True)
    sink17 = ppt.create_sink(net, junctions[16], 0.0,  name='sink 17', in_service=True)
    sink20 = ppt.create_sink(net, junctions[19], 0.0,  name='sink 20', in_service=True)
    sink21 = ppt.create_sink(net, junctions[20], 0.0,  name='sink 21', in_service=True)
    sink23 = ppt.create_sink(net, junctions[22], 0.0,  name='sink 23', in_service=True)
    sink24 = ppt.create_sink(net, junctions[23], 0.0,  name='sink 24', in_service=True)
    sink26 = ppt.create_sink(net, junctions[25], 0.0,  name='sink 26', in_service=True)
    sink27 = ppt.create_sink(net, junctions[26], 0.0,  name='sink 27', in_service=True)
    sink29 = ppt.create_sink(net, junctions[28], 0.0,  name='sink 29', in_service=True)
    sink30 = ppt.create_sink(net, junctions[29], 0.0,  name='sink 30', in_service=True)
    sink31 = ppt.create_sink(net, junctions[30], 0.0,  name='sink 31', in_service=True)
    sink32 = ppt.create_sink(net, junctions[31], 0.0,  name='sink 32', in_service=True)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink1,  profile_name='sink_m1',  data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink4,  profile_name='sink_m4',  data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink7,  profile_name='sink_m7',  data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink8,  profile_name='sink_m8',  data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink10, profile_name='sink_m10', data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink11, profile_name='sink_m11', data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink13, profile_name='sink_m13', data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink14, profile_name='sink_m14', data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink16, profile_name='sink_m16', data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink17, profile_name='sink_m17', data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink20, profile_name='sink_m20', data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink21, profile_name='sink_m21', data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink23, profile_name='sink_m23', data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink24, profile_name='sink_m24', data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink26, profile_name='sink_m26', data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink27, profile_name='sink_m27', data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink29, profile_name='sink_m29', data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink30, profile_name='sink_m30', data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink31, profile_name='sink_m31', data_source=sink_ds)
    ConstControl(net, element='sink', variable='mdot_kg_per_s', element_index=sink32, profile_name='sink_m32', data_source=sink_ds)

    ids = {
        'mass_storage1': mass_storage1, 'CHP_thermal': CHP_thermal,
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
    sink_ds = DFData(sink_m_profile.iloc[0: simulation_hours])
    CHP_thermal_input = pd.read_csv('./data/3_days_test_without_actions/output_writer/electric_net/res_ext_grid/p_mw.csv')
    CHP_thermal_input_ds = DFData(CHP_thermal_input.iloc[0: simulation_hours])
    thermal_net, ids = create_thermal_env(sink_ds, CHP_thermal_input_ds)
    owr = ow.OutputWriter(thermal_net, output_path="./data/3_days_test_without_actions/output_writer/thermal_net", output_file_type=".csv", csv_separator=',')
    owr.remove_log_variable('res_bus', 'vm_pu')
    owr.remove_log_variable('res_line', 'loading_percent')
    owr.log_variable('res_ext_grid', 'mdot_kg_per_s')
    owr.log_variable('res_sink', 'mdot_kg_per_s')
    owr.log_variable('res_source', 'mdot_kg_per_s')
    ts.run_timeseries(thermal_net, time_steps=simulation_hours, continue_on_divergence=False)

    #ppt.plotting.simple_plot(thermal_net, plot_sinks=True, plot_sources=True, pump_size=0.5)
