import pandas as pd
from parameters import *
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

# Create profile
def reindex_normalization_interpolation(df_PV, df_Wind, df_Electricity_price, df_Gas_price, df_load, df_Cornell, df_Cornell_transform_map):
    #print(f'maximum value before normalization: {df_PV["electricity"].max()},\n {df_Wind["electricity"].max()},\n {df_Electricity_price.max()},\n {df_load.max()},\n {df_sink.max()}\n')

    df_PV = pd.pivot_table(df_PV, values='electricity', index=['local_time'], sort=False)
    df_Wind = pd.pivot_table(df_Wind, values='electricity', index=['local_time'], sort=False)
    df_Electricity_price = pd.pivot_table(df_Electricity_price, values='LBMP', index=['Date'], columns=['Zone'], sort=False)
    df_Gas_price.index = pd.Series([0*24, 31*24, 58*24, 90*24, 120*24, 151*24, 181*24, 212*24, 243*24, 273*24, 304*24, 334*24])
    df_Gas_price = df_Gas_price.reindex(index=range(365 * 24), method='ffill')
    df_load = pd.pivot_table(df_load, values='Load', index=['Date'], columns=['Zone'], sort=False)

    pdate = pd.date_range(start='2021-01-01 00:00:00', end='2021-12-31 23:00:00', freq='H')

    df_PV = df_PV.set_index(pd.to_datetime(df_PV.index))
    df_PV = df_PV.reindex(pdate)
    df_PV = df_PV.interpolate()
    df_Wind = df_Wind.set_index(pd.to_datetime(df_Wind.index))
    df_Wind = df_Wind.reindex(pdate)
    df_Wind = df_Wind.interpolate()
    df_Electricity_price = df_Electricity_price.set_index(pd.to_datetime(df_Electricity_price.index))
    df_Electricity_price = df_Electricity_price.reindex(pdate)
    df_Electricity_price = df_Electricity_price.interpolate()

    df_load = df_load.set_index(pd.to_datetime(df_load.index))
    df_load = df_load.reindex(pdate)
    df_load_nan = df_load[df_load['CAPITL'].isnull().values == True]
    df_load = df_load.interpolate()
    for i in df_load_nan.index:
        for j in ['CAPITL', 'CENTRL', 'DUNWOD', 'GENESE', 'HUD VL', 'LONGIL', 'MHK VL', 'MILLWD', 'N.Y.C.']:
            df_load[j][i] = df_load[j][i] * random.uniform(0.8, 1.2)

    df_Gas_price.index = pdate
    df_Cornell.index = pdate
    df_Cornell_transform_map.index = pdate

    # scale values
    for df in [df_PV, df_Wind, df_Electricity_price, df_Gas_price, df_load, df_Cornell]:
        #print(df.max())
        df /= df.max()

    return df_PV, df_Wind, df_Electricity_price, df_Gas_price, df_load, df_Cornell, df_Cornell_transform_map

def multiply_by_max(df_PV, df_Wind, df_Electricity_price, df_Gas_price, df_load, df_Cornell, df_Cornell_transform_map):
    df_PV['electricity'] =  df_PV['electricity'] * PV1_max_p
    df_Wind['electricity'] = df_Wind['electricity'] * Wind1_max_p
    df_Electricity_price = pd.DataFrame({
        'buy_price': df_Electricity_price['N.Y.C.'] * C_Electricity_price_max,
        'sell_price': df_Electricity_price['N.Y.C.'] * C_Electricity_price_max * 0.5
    })
    df_Gas_price = pd.DataFrame({
        'price': df_Gas_price['U.S. Price of Natural Gas Delivered to Residential Consumers (Dollars per Thousand Cubic Feet)'] * C_Gas_price_max
    })
    df_load_1 = pd.DataFrame({
        'load1':  df_load['CAPITL'] * load1_max_p,
        'load4':  df_load['CENTRL'] * load2_max_p,
        'load5':  df_load['DUNWOD'] * load3_max_p,
        'load7':  df_load['GENESE'] * load4_max_p,
        'load9':  df_load['HUD VL'] * load5_max_p,
        'load10': df_load['LONGIL'] * load6_max_p,
        'load11': df_load['MHK VL'] * load7_max_p,
        'load12': df_load['MILLWD'] * load8_max_p,
        'load14': df_load['N.Y.C.'] * load9_max_p,
        'load15': df_load['NORTH']  * load10_max_p,
        'load16': df_load['WEST']   * load11_max_p,
    })

    for i in range(len(df_load)):
        for j in ['CAPITL', 'CENTRL', 'DUNWOD', 'GENESE', 'HUD VL', 'LONGIL', 'MHK VL', 'MILLWD', 'N.Y.C.']:
            df_load[j][i] = df_load[j][i] * random.uniform(0.8, 1.2)

    df_load_2 = pd.DataFrame({
        'load18': df_load['CAPITL'] * load12_max_p,
        'load19': df_load['CENTRL'] * load13_max_p,
        'load23': df_load['DUNWOD'] * load14_max_p,
        'load24': df_load['GENESE'] * load15_max_p,
        'load26': df_load['HUD VL'] * load16_max_p,
        'load28': df_load['LONGIL'] * load17_max_p,
        'load30': df_load['MHK VL'] * load18_max_p,
        'load31': df_load['MILLWD'] * load19_max_p,
        'load32': df_load['N.Y.C.'] * load20_max_p
    })

    df_load = pd.merge(df_load_1, df_load_2, left_index=True, right_index=True)

    df_sink_p_1 = pd.DataFrame({
        'sink_p1': df_Cornell['Type A MWth'] * sink1_max_p,
        'sink_p4': df_Cornell['Type B MWth'] * sink2_max_p,
        'sink_p7': df_Cornell['Type C MWth'] * sink3_max_p,
    })

    df_sink_m_1 = pd.DataFrame({
        'sink_m1': df_sink_p_1['sink_p1'] * df_Cornell_transform_map['TypeA_PtoM'] / 60 / 60,
        'sink_m4': df_sink_p_1['sink_p4'] * df_Cornell_transform_map['TypeB_PtoM'] / 60 / 60,
        'sink_m7': df_sink_p_1['sink_p7'] * df_Cornell_transform_map['TypeC_PtoM'] / 60 / 60,
    })

    for i in range(len(df_Cornell)):
        for j in ['Type A MWth', 'Type B MWth', 'Type C MWth']:
            df_Cornell[j][i] = df_Cornell[j][i] * random.uniform(0.8, 1.2)

    df_sink_p_2 = pd.DataFrame({
        'sink_p8':  df_Cornell['Type A MWth'] * sink4_max_p,
        'sink_p10': df_Cornell['Type B MWth'] * sink5_max_p,
        'sink_p11': df_Cornell['Type C MWth'] * sink6_max_p,
    })
    
    df_sink_m_2 = pd.DataFrame({
        'sink_m8' : df_sink_p_2['sink_p8']  * df_Cornell_transform_map['TypeA_PtoM'] / 60 / 60,
        'sink_m10': df_sink_p_2['sink_p10'] * df_Cornell_transform_map['TypeB_PtoM'] / 60 / 60,
        'sink_m11': df_sink_p_2['sink_p11'] * df_Cornell_transform_map['TypeC_PtoM'] / 60 / 60,
    })

    for i in range(len(df_Cornell)):
        for j in ['Type A MWth', 'Type B MWth', 'Type C MWth']:
            df_Cornell[j][i] = df_Cornell[j][i] * random.uniform(0.8, 1.2)

    df_sink_p_3 = pd.DataFrame({
        'sink_p13': df_Cornell['Type A MWth'] * sink7_max_p,
        'sink_p14': df_Cornell['Type B MWth'] * sink8_max_p,
        'sink_p16': df_Cornell['Type C MWth'] * sink9_max_p,
    })
    
    df_sink_m_3 = pd.DataFrame({
        'sink_m13': df_sink_p_3['sink_p13'] * df_Cornell_transform_map['TypeA_PtoM'] / 60 / 60,
        'sink_m14': df_sink_p_3['sink_p14'] * df_Cornell_transform_map['TypeB_PtoM'] / 60 / 60,
        'sink_m16': df_sink_p_3['sink_p16'] * df_Cornell_transform_map['TypeC_PtoM'] / 60 / 60,
    })

    for i in range(len(df_Cornell)):
        for j in ['Type A MWth', 'Type B MWth', 'Type C MWth']:
            df_Cornell[j][i] = df_Cornell[j][i] * random.uniform(0.8, 1.2)

    df_sink_p_4 = pd.DataFrame({
        'sink_p17': df_Cornell['Type A MWth'] * sink10_max_p,
        'sink_p20': df_Cornell['Type B MWth'] * sink11_max_p,
        'sink_p21': df_Cornell['Type C MWth'] * sink12_max_p,
    })
    
    df_sink_m_4 = pd.DataFrame({
        'sink_m17': df_sink_p_4['sink_p17'] * df_Cornell_transform_map['TypeA_PtoM'] / 60 / 60,
        'sink_m20': df_sink_p_4['sink_p20'] * df_Cornell_transform_map['TypeB_PtoM'] / 60 / 60,
        'sink_m21': df_sink_p_4['sink_p21'] * df_Cornell_transform_map['TypeC_PtoM'] / 60 / 60,
    })

    for i in range(len(df_Cornell)):
        for j in ['Type A MWth', 'Type B MWth', 'Type C MWth']:
            df_Cornell[j][i] = df_Cornell[j][i] * random.uniform(0.8, 1.2)

    df_sink_p_5 = pd.DataFrame({
        'sink_p23': df_Cornell['Type A MWth'] * sink13_max_p,
        'sink_p24': df_Cornell['Type B MWth'] * sink14_max_p,
        'sink_p26': df_Cornell['Type C MWth'] * sink15_max_p,
    })
    
    df_sink_m_5 = pd.DataFrame({
        'sink_m23': df_sink_p_5['sink_p23'] * df_Cornell_transform_map['TypeA_PtoM'] / 60 / 60,
        'sink_m24': df_sink_p_5['sink_p24'] * df_Cornell_transform_map['TypeB_PtoM'] / 60 / 60,
        'sink_m26': df_sink_p_5['sink_p26'] * df_Cornell_transform_map['TypeC_PtoM'] / 60 / 60,
    })

    for i in range(len(df_Cornell)):
        for j in ['Type A MWth', 'Type B MWth', 'Type C MWth']:
            df_Cornell[j][i] = df_Cornell[j][i] * random.uniform(0.8, 1.2)

    df_sink_p_6 = pd.DataFrame({
        'sink_p27': df_Cornell['Type A MWth'] * sink16_max_p,
        'sink_p29': df_Cornell['Type B MWth'] * sink17_max_p,
        'sink_p30': df_Cornell['Type C MWth'] * sink18_max_p,
    })
    
    df_sink_m_6 = pd.DataFrame({
        'sink_m27': df_sink_p_6['sink_p27'] * df_Cornell_transform_map['TypeA_PtoM'] / 60 / 60,
        'sink_m29': df_sink_p_6['sink_p29'] * df_Cornell_transform_map['TypeB_PtoM'] / 60 / 60,
        'sink_m30': df_sink_p_6['sink_p30'] * df_Cornell_transform_map['TypeC_PtoM'] / 60 / 60,
    })

    for i in range(len(df_Cornell)):
        for j in ['Type A MWth', 'Type B MWth']:
            df_Cornell[j][i] = df_Cornell[j][i] * random.uniform(0.8, 1.2)

    df_sink_p_7 = pd.DataFrame({
        'sink_p31': df_Cornell['Type A MWth'] * sink19_max_p,
        'sink_p32': df_Cornell['Type B MWth'] * sink20_max_p,
    })
    
    df_sink_m_7 = pd.DataFrame({
        'sink_m31': df_sink_p_7['sink_p31'] * df_Cornell_transform_map['TypeA_PtoM'] / 60 / 60,
        'sink_m32': df_sink_p_7['sink_p32'] * df_Cornell_transform_map['TypeB_PtoM'] / 60 / 60
    })

    for df in [df_sink_p_2, df_sink_p_3, df_sink_p_4, df_sink_p_5, df_sink_p_6, df_sink_p_7]:
        df_sink_p_1 = pd.merge(df_sink_p_1, df, left_index=True, right_index=True)

    for df in [df_sink_m_2, df_sink_m_3, df_sink_m_4, df_sink_m_5, df_sink_m_6, df_sink_m_7]:
        df_sink_m_1 = pd.merge(df_sink_m_1, df, left_index=True, right_index=True)

    df_sink_m_1 = -df_sink_m_1
    df_sink_m_1[df_sink_m_1 > 0] = 0

    return df_PV, df_Wind, df_Electricity_price, df_Gas_price, df_load, df_sink_p_1, df_sink_m_1

def save_profile(PV_profile, Wind_profile, Electricity_price_profile, Gas_price_profile, load_profile, sink_p_profile, sink_m_profile, df_Cornell_transform_map):
    PV_profile.to_csv('./data/profile/PV_profile.csv')
    Wind_profile.to_csv('./data/profile/Wind_profile.csv')
    Electricity_price_profile.to_csv('./data/profile/Electricity_price_profile.csv')
    Gas_price_profile.to_csv('./data/profile/Gas_price_profile.csv')
    load_profile.to_csv('./data/profile/load_profile.csv')
    sink_p_profile.to_csv('./data/profile/sink_p_profile.csv')
    sink_m_profile.to_csv('./data/profile/sink_m_profile.csv')
    df_Cornell_transform_map.to_csv('./data/Cornell_transform_map.csv')

# Plot profile
def plot_profile(PV_profile, Wind_profile, Electricity_price_profile, Gas_price_profile, load_profile, sink_p_profile, sink_m_profile, plot_days):
    plot_days = plot_days * 24
    x = range(0, plot_days)

    plt.figure(1)
    PV_plot_data = PV_profile['electricity'][:plot_days]
    plt.bar(x, PV_plot_data, label='PV')
    plt.xlabel('t (hour)')
    plt.ylabel('MWe')
    plt.legend(loc='upper right')
    plt.title('PV Power Generation')
    print("fig1 done!")

    plt.figure(2)
    Wind_plot_data = Wind_profile['electricity'][:plot_days]
    plt.bar(x, Wind_plot_data, label='Wind Turbine')
    plt.xlabel('t (hour)')
    plt.ylabel('MWe')
    plt.legend(loc='upper right')
    plt.title('Wind Turbine Power Generation')
    print("fig2 done!")

    plt.figure(3)
    Electricity_buy_price_plot_data = Electricity_price_profile['buy_price'].values[:plot_days] * 963.91 # N.Y.C.'s Electricity price df.max()
    plt.plot(x, Electricity_buy_price_plot_data, label='Buy_Price')
    Electricity_sell_price_plot_data = Electricity_price_profile['sell_price'].values[:plot_days] * 963.91  # N.Y.C.'s Electricity price df.max()
    plt.plot(x, Electricity_sell_price_plot_data, label='Sell_Price')
    plt.xlabel('t (hour)')
    plt.ylabel('USD/MWeh')
    plt.legend(loc='upper right')
    plt.title('Price of Electricity')
    print("fig3 done!")

    plt.figure(4)
    Gas_price_plot_data = Gas_price_profile['price'].values[:plot_days] * 23.01 * 28.317 # Gas price df.max(), mcf to m³
    plt.plot(x, Gas_price_plot_data, label='Price')
    plt.xlabel('t (hour)')
    plt.ylabel('USD/m³')
    plt.legend(loc='upper right')
    plt.title('Price of Natural Gas')
    print("fig4 done!")

    plt.figure(5)
    load1_plot_data = load_profile['load1'].values[:plot_days]
    plt.bar(x, load1_plot_data, label='load1')
    load_temp_plot_data = load1_plot_data
    for i in ['load4', 'load5', 'load7', 'load9', 'load10', 'load11', 'load12', 'load14', 'load15', 'load16', 'load18', 'load19', 'load23', 'load24', 'load26', 'load28', 'load30', 'load31', 'load32']:
        locals()[i + '_plot_data'] = load_profile[i].values[:plot_days]
        plt.bar(x, locals()[i + '_plot_data'], bottom=load_temp_plot_data, label=i)
        load_temp_plot_data = load_temp_plot_data + locals()[i + '_plot_data']
    plt.xlabel('t (hour)')
    plt.ylabel('MWe')
    plt.legend(loc='upper right')
    plt.title('Total Load')
    print("fig5 done!")

    plt.figure(6)
    sink_p1_plot_data = sink_p_profile['sink_p1'].values[:plot_days]
    plt.bar(x, sink_p1_plot_data, label='sink_p1')
    sink_p_temp_plot_data = sink_p1_plot_data
    for i in ['sink_p4', 'sink_p7', 'sink_p8', 'sink_p10', 'sink_p11', 'sink_p13', 'sink_p14', 'sink_p16', 'sink_p17', 'sink_p20', 'sink_p21', 'sink_p23', 'sink_p24', 'sink_p26', 'sink_p27', 'sink_p29', 'sink_p30', 'sink_p31', 'sink_p32']:
        locals()[i + '_plot_data'] = sink_p_profile[i].values[:plot_days]
        plt.bar(x, locals()[i + '_plot_data'], bottom=sink_p_temp_plot_data, label=i)
        sink_p_temp_plot_data = sink_p_temp_plot_data + locals()[i + '_plot_data']
    plt.xlabel('t (hour)')
    plt.ylabel('MWt')
    plt.legend(loc='upper right')
    plt.title('Total Sink Power')
    print("fig6 done!")
    
    plt.figure(7)
    sink_m1_plot_data = sink_m_profile['sink_m1'].values[:plot_days]
    plt.bar(x, sink_m1_plot_data, label='sink_m1')
    sink_m_temp_plot_data = sink_m1_plot_data
    for i in ['sink_m4', 'sink_m7', 'sink_m8', 'sink_m10', 'sink_m11', 'sink_m13', 'sink_m14', 'sink_m16', 'sink_m17', 'sink_m20', 'sink_m21', 'sink_m23', 'sink_m24', 'sink_m26', 'sink_m27', 'sink_m29', 'sink_m30', 'sink_m31', 'sink_m32']:
        locals()[i + '_plot_data'] = sink_m_profile[i].values[:plot_days]
        plt.bar(x, locals()[i + '_plot_data'], bottom=sink_m_temp_plot_data, label=i)
        sink_m_temp_plot_data = sink_m_temp_plot_data + locals()[i + '_plot_data']
    plt.xlabel('t (hour)')
    plt.ylabel('kg/s')
    plt.legend(loc='upper right')
    plt.title('Total Sink Mass')
    print("fig7 done!")

    plt.show()

def create_MtoP_and_PtoM_transform_map(df_Cornell):
    df_Cornell_transform_map = pd.DataFrame({
        'TypeA_MtoP': df_Cornell['Type A MWth'] / df_Cornell['Type A kg'],
        'TypeA_PtoM': df_Cornell['Type A kg'] / df_Cornell['Type A MWth'],
        'TypeB_MtoP': df_Cornell['Type B MWth'] / df_Cornell['Type B kg'],
        'TypeB_PtoM': df_Cornell['Type B kg'] / df_Cornell['Type B MWth'],
        'TypeC_MtoP': df_Cornell['Type C MWth'] / df_Cornell['Type C kg'],
        'TypeC_PtoM': df_Cornell['Type C kg'] / df_Cornell['Type C MWth'],
        'Totals_MtoP': df_Cornell['Steam Totals MWth'] / df_Cornell['Steam Totals kg'],
        'Totals_PtoM': df_Cornell['Steam Totals kg'] / df_Cornell['Steam Totals MWth'],
    })

    return df_Cornell_transform_map


if __name__ == '__main__':
    df_PV = pd.read_csv('data/raw_data/renewable_energy_source/ninja_pv_40.8667_-72.8500_2021.csv', usecols=['local_time', 'electricity'], header=3)
    df_Wind = pd.read_csv('data/raw_data/renewable_energy_source/ninja_wind_43.7853_-75.5753_2021.csv', usecols=['local_time', 'electricity'], header=3)
    df_Electricity_price = pd.read_csv('data/raw_data/20210101-20211231 NYISO Actual Energy Price.csv', usecols=['Date', 'LBMP', 'Zone'])
    df_Gas_price = pd.read_csv('data/raw_data/N3010US3m.csv', usecols=['U.S. Price of Natural Gas Delivered to Residential Consumers (Dollars per Thousand Cubic Feet)'], header=2)
    df_load = pd.read_csv('data/raw_data/20210101-20211231 NYISO Hourly Actual Load.csv', usecols=['Date', 'Load', 'Zone'])
    df_Cornell = pd.read_csv('data/raw_data/Cornell_Hourly_Steam_Data_FY17 (for upload).csv', usecols=['Total\nType A\nBuildings lbs steam', 'Total\nType B\nBuildings lbs steam', 'Total\nType C\nBuildings lbs steam', 'Steam Totals lbs steam', 'Type A MWth', 'Type B MWth', 'Type C MWth', 'Steam Totals MWth'], thousands=",")
    for i in ['Total\nType A\nBuildings lbs steam', 'Total\nType B\nBuildings lbs steam', 'Total\nType C\nBuildings lbs steam', 'Steam Totals lbs steam']:
        df_Cornell[i] = df_Cornell[i] * 0.45359237
    df_Cornell = df_Cornell.rename({'Total\nType A\nBuildings lbs steam': 'Type A kg', 'Total\nType B\nBuildings lbs steam': 'Type B kg', 'Total\nType C\nBuildings lbs steam': 'Type C kg', 'Steam Totals lbs steam': 'Steam Totals kg'}, axis='columns')
    '''
    df_sink_p = pd.read_csv('data/raw_data/Cornell_Hourly_Steam_Data_FY17 (for upload).csv', usecols=['Type A MWth', 'Type B MWth', 'Type C MWth'])
    df_sink_m = pd.read_csv('data/raw_data/Cornell_Hourly_Steam_Data_FY17 (for upload).csv', usecols=['Total\nType A\nBuildings lbs steam', 'Total\nType B\nBuildings lbs steam', 'Total\nType C\nBuildings lbs steam'], thousands=",")
    '''
    df_Cornell_transform_map = create_MtoP_and_PtoM_transform_map(df_Cornell)
    df_PV, df_Wind, df_Electricity_price, df_Gas_price, df_load, df_Cornell, df_Cornell_transform_map = reindex_normalization_interpolation(df_PV, df_Wind, df_Electricity_price, df_Gas_price, df_load, df_Cornell, df_Cornell_transform_map)
    PV_profile, Wind_profile, Electricity_price_profile, Gas_price_profile, load_profile, sink_p_profile, sink_m_profile = multiply_by_max(df_PV, df_Wind, df_Electricity_price, df_Gas_price, df_load, df_Cornell, df_Cornell_transform_map)
    save_profile(PV_profile, Wind_profile, Electricity_price_profile, Gas_price_profile, load_profile, sink_p_profile, sink_m_profile, df_Cornell_transform_map)
    print("profile done!")

    plot_profile(PV_profile, Wind_profile, Electricity_price_profile, Gas_price_profile, load_profile, sink_p_profile, sink_m_profile, 365)

