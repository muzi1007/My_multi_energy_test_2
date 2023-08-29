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
    sink_m_profile = pd.read_csv('./data/profile/sink_m_profile.csv')

    electricity_res_ext_grid_record = []
    thermal_res_ext_grid_record = []
    CHP_electricity_record = []
    CHP_thermal_record = []

    for i in range(0, simulation_hours):
        print('hour ', i)
        electric_net, ids = create_electric_env(PV_profile['electricity'], Wind_profile['electricity'], load_profile, E_Bat_c_max_list, i)
        ppe.runpp(electric_net, numba=False)
        #print(net.res_ext_grid)
        electricity_res_ext_grid_record.append(electric_net.res_ext_grid.p_mw.values[0])
        CHP_electricity_record.append(electric_net.res_ext_grid.p_mw.values[1])
        CHP_thermal_input = electric_net.res_ext_grid.p_mw[1] * 1 * 491.9137466307278 / 60 / 60
        CHP_thermal_record.append(CHP_thermal_input)
        thermal_net, ids = create_thermal_env(sink_m_profile, Th_Bat_d_min_list, -CHP_thermal_input, i)
        ppt.pipeflow(thermal_net)
        thermal_res_ext_grid_record.append(thermal_net.res_ext_grid.mdot_kg_per_s.values[0])
    print("simulate done!")

    #ppe.plotting.simple_plot(electric_net, plot_gens=True, plot_sgens=True, plot_line_switches=True, plot_loads=True)
    #ppe.plotting.to_html(electric_net, 'env/electric_net_test.html')

    plt.figure(1)
    plt.plot(range(0, simulation_hours), electricity_res_ext_grid_record)
    plt.xlabel('hours (hr)', size=12)
    plt.ylabel('Electricity (MWe)', size=12)
    plt.legend(['res_ext_grid'], loc='upper right')
    plt.title('Electricity from external grid')

    plt.figure(2)
    plt.plot(range(0, simulation_hours), thermal_res_ext_grid_record)
    plt.xlabel('hours (hr)', size=12)
    plt.ylabel('Mass (kg/s)', size=12)
    plt.legend(['res_ext_grid'], loc='upper right')
    plt.title('Mass from external grid')

    plt.figure(3)
    plt.plot(range(0, simulation_hours), CHP_electricity_record)
    plt.xlabel('hours (hr)', size=12)
    plt.ylabel('Electricity (MWe)', size=12)
    plt.legend(['CHP_electricity'], loc='upper right')
    plt.title('Electricity from CHP')

    plt.figure(4)
    plt.plot(range(0, simulation_hours), CHP_thermal_record)
    plt.xlabel('hours (hr)', size=12)
    plt.ylabel('Mass (kg/s)', size=12)
    plt.legend(['CHP_thermal'], loc='upper right')
    plt.title('Mass from CHP')

    plt.show()


