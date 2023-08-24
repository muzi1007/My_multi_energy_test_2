import random

# load max power (MWe)
load1_max_p = 1.85
load2_max_p = 2
load3_max_p = 3.5
load4_max_p = 1.25
load5_max_p = 2.9
load6_max_p = 1.2
load7_max_p = 2.65
load8_max_p = 3.565
load9_max_p = 1.605
load10_max_p = 2.45
load11_max_p = 1.55
load12_max_p = 2.85
load13_max_p = 3.05
load14_max_p = 1.35
load15_max_p = 2.525
load16_max_p = 1.555
load17_max_p = 2.85
load18_max_p = 3.95
load19_max_p = 1
load20_max_p = 2.525

load_max_p_list = [load1_max_p, load2_max_p, load3_max_p, load4_max_p, load5_max_p, load6_max_p, load7_max_p, load8_max_p, load9_max_p,
                    load10_max_p, load11_max_p, load12_max_p, load13_max_p, load14_max_p, load15_max_p, load16_max_p, load17_max_p,
                    load18_max_p, load19_max_p, load20_max_p]

# sink max mass (MWt)
# Wt = 1012 * \dot{m_q} * (T_s - T_0), Watt (Wt) = Specific heat of air * vector of the mass flow rate (kg/s) * Temperature change (C)
sink1_max_p = 1.85
sink2_max_p = 2.95
sink3_max_p = 1.45
sink4_max_p = 1.45
sink5_max_p = 2.565
sink6_max_p = 1.125
sink7_max_p = 1.875
sink8_max_p = 2.55
sink9_max_p = 1.125
sink10_max_p = 1.255
sink11_max_p = 2.95
sink12_max_p = 1.25
sink13_max_p = 1.255
sink14_max_p = 2.65
sink15_max_p = 1.95
sink16_max_p = 1.25
sink17_max_p = 2.855
sink18_max_p = 1.75
sink19_max_p = 1.235
sink20_max_p = 2.225

sink_max_p_list = [sink1_max_p, sink2_max_p, sink3_max_p, sink4_max_p, sink5_max_p, sink6_max_p, sink7_max_p, sink8_max_p, sink9_max_p,
                    sink10_max_p, sink11_max_p, sink12_max_p, sink13_max_p, sink14_max_p, sink15_max_p, sink16_max_p, sink17_max_p,
                    sink18_max_p, sink19_max_p, sink20_max_p]

# PV max power (MW)
PV1_max_p = 0.99
PV_max_p_list = [PV1_max_p]

# Wind turbine max power (MW)
Wind1_max_p = 12
Wind_max_p_list = [Wind1_max_p]

# Battery capacity (MWh), SoC min/max, charge/discharge min/max (MW)
E_Bat1_E = 2
E_Bat_E_list = [E_Bat1_E]

E_SoC_max = 0.9
E_SoC_min = 0.1

E_Bat1_c_max = 1. # Original charge/discharge are 0.8, -0.8
E_Bat1_d_min = -1.
E_Bat_c_max_list = [E_Bat1_c_max]
E_Bat_d_min_list = [E_Bat1_d_min]

# Thermal battery capacity (?), SoC min/max, charge/discharge min/max (?)
Th_Bat1_E = 1
Th_Bat_E_list = [Th_Bat1_E]

Th_SoC_max = 0.9
Th_SoC_min = 0.1

Th_Bat1_c_max = 1.
Th_Bat1_d_min = -1.
Th_Bat_c_max_list = [Th_Bat1_c_max]
Th_Bat_d_min_list = [Th_Bat1_d_min]

# Cost
C_Electricity_price_max = 1
C_Gas_price_max = 1