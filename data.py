import pandas as pd
from parameters import *
import matplotlib.pyplot as plt

# Create profile

def normalization_interpolation(df_PV, df_Wind, df_Electricity_price, df_Gas_price, df_load, df_sink):
    #print(f'maximum value before normalization: {df_PV["electricity"].max()},\n {df_Wind["electricity"].max()},\n {df_Electricity_price.max()},\n {df_load.max()},\n {df_sink.max()}\n')

    df_PV = pd.pivot_table(df_PV, values='electricity', index=['local_time'], sort=False)
    df_Wind = pd.pivot_table(df_Wind, values='electricity', index=['local_time'], sort=False)
    df_Electricity_price = pd.pivot_table(df_Electricity_price, values='LBMP', index=['Date'], columns=['Zone'], sort=False)
    df_Gas_price.index = pd.Series([0*24, 31*24, 58*24, 90*24, 120*24, 151*24, 181*24, 212*24, 243*24, 273*24, 304*24, 334*24])
    df_Gas_price = df_Gas_price.reindex(index=range(365 * 24), method='ffill')
    df_load = pd.pivot_table(df_load, values='Load', index=['Date'], columns=['Zone'], sort=False)

    pdate = pd.date_range(start='2019-01-01 00:00:00', end='2019-12-31 23:00:00', freq='H')

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
        df_load['CAPITL'][i] = df_load['CAPITL'][i] * random.uniform(0.8, 1.2)
        df_load['CENTRL'][i] = df_load['CENTRL'][i] * random.uniform(0.8, 1.2)
        df_load['DUNWOD'][i] = df_load['DUNWOD'][i] * random.uniform(0.8, 1.2)
        df_load['GENESE'][i] = df_load['GENESE'][i] * random.uniform(0.8, 1.2)
        df_load['HUD VL'][i] = df_load['HUD VL'][i] * random.uniform(0.8, 1.2)
        df_load['LONGIL'][i] = df_load['LONGIL'][i] * random.uniform(0.8, 1.2)
        df_load['MHK VL'][i] = df_load['MHK VL'][i] * random.uniform(0.8, 1.2)
        df_load['MILLWD'][i] = df_load['MILLWD'][i] * random.uniform(0.8, 1.2)
        df_load['N.Y.C.'][i] = df_load['N.Y.C.'][i] * random.uniform(0.8, 1.2)

    df_Gas_price.index = pdate
    df_sink.index = pdate

    # scale values
    for df in [df_PV, df_Wind, df_Electricity_price, df_Gas_price, df_load, df_sink]:
        #print(df.max())
        df /= df.max()

    return df_PV, df_Wind, df_Electricity_price, df_Gas_price, df_load, df_sink

def multiply_by_max(df_PV, df_Wind, df_Electricity_price, df_Gas_price, df_load, df_sink):
    df_PV['electricity'] =  df_PV['electricity'] * PV1_max_p
    df_Wind['electricity'] = df_Wind['electricity'] * Wind1_max_p
    df_Electricity_price = pd.DataFrame({
        'price': df_Electricity_price['N.Y.C.'] * C_Electricity_price_max
    })
    df_Gas_price = pd.DataFrame({
        'price': df_Gas_price['U.S. Price of Natural Gas Delivered to Residential Consumers (Dollars per Thousand Cubic Feet)'] * C_Gas_price_max
    })
    df_load_1 = pd.DataFrame({
        'load1': df_load['CAPITL'] * load1_max_p,
        'load4': df_load['CENTRL'] * load2_max_p,
        'load5': df_load['DUNWOD'] * load3_max_p,
        'load7': df_load['GENESE'] * load4_max_p,
        'load9': df_load['HUD VL'] * load5_max_p,
        'load10': df_load['LONGIL'] * load6_max_p,
        'load11': df_load['MHK VL'] * load7_max_p,
        'load12': df_load['MILLWD'] * load8_max_p,
        'load14': df_load['N.Y.C.'] * load9_max_p,
        'load15': df_load['NORTH'] * load10_max_p,
        'load16': df_load['WEST'] * load11_max_p,
    })

    for i in range(len(df_load)):
        df_load['CAPITL'][i] = df_load['CAPITL'][i] * random.uniform(0.8, 1.2)
        df_load['CENTRL'][i] = df_load['CENTRL'][i] * random.uniform(0.8, 1.2)
        df_load['DUNWOD'][i] = df_load['DUNWOD'][i] * random.uniform(0.8, 1.2)
        df_load['GENESE'][i] = df_load['GENESE'][i] * random.uniform(0.8, 1.2)
        df_load['HUD VL'][i] = df_load['HUD VL'][i] * random.uniform(0.8, 1.2)
        df_load['LONGIL'][i] = df_load['LONGIL'][i] * random.uniform(0.8, 1.2)
        df_load['MHK VL'][i] = df_load['MHK VL'][i] * random.uniform(0.8, 1.2)
        df_load['MILLWD'][i] = df_load['MILLWD'][i] * random.uniform(0.8, 1.2)
        df_load['N.Y.C.'][i] = df_load['N.Y.C.'][i] * random.uniform(0.8, 1.2)

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

    df_sink1 = pd.DataFrame({
        'sink1': df_sink['Type A MWth'] * sink1_max_p,
        'sink4': df_sink['Type B MWth'] * sink2_max_p,
        'sink7': df_sink['Type C MWth'] * sink3_max_p,
    })

    for i in range(len(df_sink)):
        df_sink['Type A MWth'][i] = df_sink['Type A MWth'][i] * random.uniform(0.8, 1.2)
        df_sink['Type B MWth'][i] = df_sink['Type B MWth'][i] * random.uniform(0.8, 1.2)
        df_sink['Type C MWth'][i] = df_sink['Type C MWth'][i] * random.uniform(0.8, 1.2)

    df_sink_2 = pd.DataFrame({
        'sink8': df_sink['Type A MWth'] * sink4_max_p,
        'sink10': df_sink['Type B MWth'] * sink5_max_p,
        'sink11': df_sink['Type C MWth'] * sink6_max_p,
    })

    for i in range(len(df_sink)):
        df_sink['Type A MWth'][i] = df_sink['Type A MWth'][i] * random.uniform(0.8, 1.2)
        df_sink['Type B MWth'][i] = df_sink['Type B MWth'][i] * random.uniform(0.8, 1.2)
        df_sink['Type C MWth'][i] = df_sink['Type C MWth'][i] * random.uniform(0.8, 1.2)

    df_sink_3 = pd.DataFrame({
        'sink13': df_sink['Type A MWth'] * sink7_max_p,
        'sink14': df_sink['Type B MWth'] * sink8_max_p,
        'sink16': df_sink['Type C MWth'] * sink9_max_p,
    })

    for i in range(len(df_sink)):
        df_sink['Type A MWth'][i] = df_sink['Type A MWth'][i] * random.uniform(0.8, 1.2)
        df_sink['Type B MWth'][i] = df_sink['Type B MWth'][i] * random.uniform(0.8, 1.2)
        df_sink['Type C MWth'][i] = df_sink['Type C MWth'][i] * random.uniform(0.8, 1.2)

    df_sink_4 = pd.DataFrame({
        'sink17': df_sink['Type A MWth'] * sink10_max_p,
        'sink20': df_sink['Type B MWth'] * sink11_max_p,
        'sink21': df_sink['Type C MWth'] * sink12_max_p,
    })

    for i in range(len(df_sink)):
        df_sink['Type A MWth'][i] = df_sink['Type A MWth'][i] * random.uniform(0.8, 1.2)
        df_sink['Type B MWth'][i] = df_sink['Type B MWth'][i] * random.uniform(0.8, 1.2)
        df_sink['Type C MWth'][i] = df_sink['Type C MWth'][i] * random.uniform(0.8, 1.2)

    df_sink_5 = pd.DataFrame({
        'sink23': df_sink['Type A MWth'] * sink13_max_p,
        'sink24': df_sink['Type B MWth'] * sink14_max_p,
        'sink26': df_sink['Type C MWth'] * sink15_max_p,
    })

    for i in range(len(df_sink)):
        df_sink['Type A MWth'][i] = df_sink['Type A MWth'][i] * random.uniform(0.8, 1.2)
        df_sink['Type B MWth'][i] = df_sink['Type B MWth'][i] * random.uniform(0.8, 1.2)
        df_sink['Type C MWth'][i] = df_sink['Type C MWth'][i] * random.uniform(0.8, 1.2)

    df_sink_6 = pd.DataFrame({
        'sink27': df_sink['Type A MWth'] * sink16_max_p,
        'sink29': df_sink['Type B MWth'] * sink17_max_p,
        'sink30': df_sink['Type C MWth'] * sink18_max_p,
    })

    for i in range(len(df_sink)):
        df_sink['Type A MWth'][i] = df_sink['Type A MWth'][i] * random.uniform(0.8, 1.2)
        df_sink['Type B MWth'][i] = df_sink['Type B MWth'][i] * random.uniform(0.8, 1.2)

    df_sink_7 = pd.DataFrame({
        'sink31': df_sink['Type A MWth'] * sink19_max_p,
        'sink32': df_sink['Type B MWth'] * sink20_max_p,
    })

    for df in [df_sink_2, df_sink_3, df_sink_4, df_sink_5, df_sink_6, df_sink_7]:
        df_sink1 = pd.merge(df_sink1, df, left_index=True, right_index=True)

    return df_PV, df_Wind, df_Electricity_price, df_Gas_price, df_load, df_sink1

def save_profile(PV_profile, Wind_profile, Electricity_price_profile, Gas_price_profile, load_profile, sink_profile):
    PV_profile.to_csv('./data/profile/PV_profile.csv')
    Wind_profile.to_csv('./data/profile/Wind_profile.csv')
    Electricity_price_profile.to_csv('./data/profile/Electricity_price_profile.csv')
    Gas_price_profile.to_csv('./data/profile/Gas_price_profile.csv')
    load_profile.to_csv('./data/profile/load_profile.csv')
    sink_profile.to_csv('./data/profile/sink_profile.csv')

# Plot profile

def plot_profile(PV_profile, Wind_profile, Electricity_price_profile, Gas_price_profile, load_profile, sink_profile, plot_days):
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
    Electricity_price_plot_data = Electricity_price_profile['price'].values[:plot_days] * 963.91 # N.Y.C.'s Electricity price df.max()
    plt.plot(x, Electricity_price_plot_data, label='Price')
    plt.xlabel('t (hour)')
    plt.ylabel('USD/MWeh')
    plt.legend(loc='upper right')
    plt.title('Price of Electricity')
    print("fig3 done!")

    plt.figure(4)
    Gas_price_plot_data = Gas_price_profile['price'].values[:plot_days] * 18.37 * 28.317 # Gas price df.max(), mcf to m³
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
    sink1_plot_data = sink_profile['sink1'].values[:plot_days]
    plt.bar(x, sink1_plot_data, label='sink1')
    sink_temp_plot_data = sink1_plot_data
    for i in ['sink4', 'sink7', 'sink8', 'sink10', 'sink11', 'sink13', 'sink14', 'sink16', 'sink17', 'sink20', 'sink21', 'sink23', 'sink24', 'sink26', 'sink27', 'sink29', 'sink30', 'sink31', 'sink32']:
        locals()[i + '_plot_data'] = sink_profile[i].values[:plot_days]
        plt.bar(x, locals()[i + '_plot_data'], bottom=sink_temp_plot_data, label=i)
        sink_temp_plot_data = sink_temp_plot_data + locals()[i + '_plot_data']
    plt.xlabel('t (hour)')
    plt.ylabel('MWt')
    plt.legend(loc='upper right')
    plt.title('Total Sink')
    print("fig6 done!")

    plt.show()


if __name__ == '__main__':
    df_PV = pd.read_csv('data/raw_data/renewable_energy_source/ninja_pv_40.8667_-72.8500.csv', usecols=['local_time', 'electricity'], header=3)
    df_Wind = pd.read_csv('data/raw_data/renewable_energy_source/ninja_wind_43.7853_-75.5753.csv', usecols=['local_time', 'electricity'], header=3)
    df_Electricity_price = pd.read_csv('data/raw_data/20190101-20191231 NYISO Actual Energy Price.csv', usecols=['Date', 'LBMP', 'Zone'])
    df_Gas_price = pd.read_csv('data/raw_data/N3010US3m.csv', usecols=['U.S. Price of Natural Gas Delivered to Residential Consumers (Dollars per Thousand Cubic Feet)'], header=2)
    df_load = pd.read_csv('data/raw_data/20190101-20191231 NYISO Hourly Actual Load.csv', usecols=['Date', 'Load', 'Zone'])
    df_sink = pd.read_csv('data/raw_data/Cornell_Hourly_Steam_Data_FY17 (for upload).csv', usecols=['Type A MWth', 'Type B MWth', 'Type C MWth'])

    df_PV, df_Wind, df_Electricity_price, df_Gas_price, df_load, df_sink = normalization_interpolation(df_PV, df_Wind, df_Electricity_price, df_Gas_price, df_load, df_sink)
    PV_profile, Wind_profile, Electricity_price_profile, Gas_price_profile, load_profile, sink_profile = multiply_by_max(df_PV, df_Wind, df_Electricity_price, df_Gas_price, df_load, df_sink)
    save_profile(PV_profile, Wind_profile, Electricity_price_profile, Gas_price_profile, load_profile, sink_profile)

    print("profile done!")

    plot_profile(PV_profile, Wind_profile, Electricity_price_profile, Gas_price_profile, load_profile, sink_profile, 3)

