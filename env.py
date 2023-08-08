import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

import pandapower as ppe
import pandapipes as ppth

from electric_env import *
from thermal_env import *
from parameters import *

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
    '''
    plt.figure()
    plt.plot(range(0, simulation_hours), res_ext_grid_record)
    plt.xlabel('hours (hr)', size=12)
    plt.ylabel('Electricity (MW)', size=12)
    plt.legend(['res_ext_grid'], loc='upper right')
    plt.title('Electricity from external grid')
    '''
    #ppe.plotting.simple_plot(electric_net, plot_gens=True, plot_sgens=True, plot_line_switches=True, plot_loads=True)
    #ppe.plotting.to_html(electric_net, 'env/electric_net_test.html')

    CHP_thermal_input = electric_net.res_ext_grid.p_mw[1] * 1000000 * 3 / 1012 / 13 / 60 / 60
    thermal_net, ids = create_thermal_env(sink_mass_list, th_Bat_d_min_list, CHP_thermal_input)
    ppth.pipeflow(thermal_net)

    print("done!")