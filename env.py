import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

import pandapower as ppe
import pandapipes as ppt
from pandapower.timeseries.data_sources.frame_data import DFData
import pandapower.timeseries as tse
import pandapipes.timeseries as tst
import pandapower.timeseries.output_writer as ow

from electric_env import *
from thermal_env import *
from parameters import *

if __name__ == "__main__":
    """
    Functional Test
    """
    simulation_hours = 72
    PV_profile = pd.read_csv('./data/profile/PV_profile.csv')
    Wind_profile = pd.read_csv('./data/profile/Wind_profile.csv')
    load_profile = pd.read_csv('./data/profile/load_profile.csv')
    sink_m_profile = pd.read_csv('./data/profile/sink_m_profile.csv')
    PV_ds = DFData(PV_profile.iloc[0: simulation_hours])
    Wind_ds = DFData(Wind_profile.iloc[0: simulation_hours])
    load_ds = DFData(load_profile.iloc[0: simulation_hours])
    sink_ds = DFData(sink_m_profile.iloc[0: simulation_hours])

    electric_net, ids = create_electric_env(PV_ds, Wind_ds, load_ds)
    owr_electric = ow.OutputWriter(electric_net, output_path="./data/3_days_test_without_actions/output_writer/electric_net", output_file_type=".csv", csv_separator=',')
    owr_electric.remove_log_variable('res_bus', 'vm_pu')
    owr_electric.remove_log_variable('res_line', 'loading_percent')
    owr_electric.log_variable('res_ext_grid', 'p_mw')
    owr_electric.log_variable('res_load', 'p_mw')
    owr_electric.log_variable('res_sgen', 'p_mw')

    tse.run_timeseries(electric_net, time_steps=simulation_hours, continue_on_divergence=False)

    CHP_thermal_input = pd.read_csv('./data/3_days_test_without_actions/output_writer/electric_net/res_ext_grid/p_mw.csv') * 1 * 491.9137466307278 / 60 / 60
    CHP_thermal_input_ds = DFData(CHP_thermal_input.iloc[0: simulation_hours])
    thermal_net, ids = create_thermal_env(sink_ds, CHP_thermal_input_ds)
    owr_thermal = ow.OutputWriter(thermal_net, output_path="./data/3_days_test_without_actions/output_writer/thermal_net", output_file_type=".csv", csv_separator=',')
    owr_thermal.remove_log_variable('res_bus', 'vm_pu')
    owr_thermal.remove_log_variable('res_line', 'loading_percent')
    owr_thermal.log_variable('res_ext_grid', 'mdot_kg_per_s')
    owr_thermal.log_variable('res_sink', 'mdot_kg_per_s')
    owr_thermal.log_variable('res_source', 'mdot_kg_per_s')
    tst.run_timeseries(thermal_net, time_steps=simulation_hours, continue_on_divergence=False)