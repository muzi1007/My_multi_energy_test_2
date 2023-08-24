import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

import pandapower as ppe
import pandapipes as ppt

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
    sink_profile = pd.read_csv('./data/profile/sink_profile.csv')
    # sink_profile 還未將 Wt 換算成 kg/s

    res_ext_grid_record = []
    for i in range(0, simulation_hours):
        #print('hour ', i)
        electric_net, ids = create_electric_env(PV_profile['electricity'], Wind_profile['electricity'], load_profile, E_Bat_c_max_list, i)
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
    thermal_net, ids = create_thermal_env(sink_profile, Th_Bat_d_min_list, CHP_thermal_input) # thermal_net 還未加上 simulation_hours
    ppt.pipeflow(thermal_net)

    print("done!")