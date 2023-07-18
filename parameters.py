import random

# Load max power (MW)
'''
load1_max_p = 0.46
load2_max_p = 0.51
load3_max_p = 0.51
load4_max_p = 0.46
load5_max_p = 0.46
load6_max_p = 1.14
load7_max_p = 0.51
load8_max_p = 0.51
load9_max_p = 0.46
load10_max_p = 0.51
load11_max_p = 0.46
load12_max_p = 1.14
load13_max_p = 1.14
load14_max_p = 0.46
load15_max_p = 1.14
load16_max_p = 0.46
load17_max_p = 0.51
load18_max_p = 0.51
load19_max_p = 1.14
load20_max_p = 0.46
'''

load1_max_p = random.uniform(0.1, 1.2)
load2_max_p = random.uniform(0.1, 1.2)
load3_max_p = random.uniform(0.1, 1.2)
load4_max_p = random.uniform(0.1, 1.2)
load5_max_p = random.uniform(0.1, 1.2)
load6_max_p = random.uniform(0.1, 1.2)
load7_max_p = random.uniform(0.1, 1.2)
load8_max_p = random.uniform(0.1, 1.2)
load9_max_p = random.uniform(0.1, 1.2)
load10_max_p = random.uniform(0.1, 1.2)
load11_max_p = random.uniform(0.1, 1.2)
load12_max_p = random.uniform(0.1, 1.2)
load13_max_p = random.uniform(0.1, 1.2)
load14_max_p = random.uniform(0.1, 1.2)
load15_max_p = random.uniform(0.1, 1.2)
load16_max_p = random.uniform(0.1, 1.2)
load17_max_p = random.uniform(0.1, 1.2)
load18_max_p = random.uniform(0.1, 1.2)
load19_max_p = random.uniform(0.1, 1.2)
load20_max_p = random.uniform(0.1, 1.2)

load_max_p_list = [load1_max_p, load2_max_p, load3_max_p, load4_max_p, load5_max_p, load6_max_p, load7_max_p, load8_max_p, load9_max_p,
                    load10_max_p, load11_max_p, load12_max_p, load13_max_p, load14_max_p, load15_max_p, load16_max_p, load17_max_p,
                    load18_max_p, load19_max_p, load20_max_p]

# PV max power (MW)
PV1_max_p = 1
pv_max_p_list = [PV1_max_p]

# Wind turbine max power (MW)
Wind1_max_p = 1.2
Wind_max_p_list = [Wind1_max_p]

# Battery capacity (MWh), SoC min/max, charge/discharge min/max (MW)
Bat1_E = 2
Bat_E_list = [Bat1_E]

SoC_max = 0.9
SoC_min = 0.1

Bat1_c_max = 1. # Original charge/discharge are 0.8, -0.8
Bat1_d_min = -1.
Bat_c_max_list = [Bat1_c_max]
Bat_d_min_list = [Bat1_d_min]
