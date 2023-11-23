import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # pycharm bug
import matplotlib.pyplot as plt

import calendar
import datetime
from tqdm import tqdm

def randomized_raw(data_name):
    data = pd.read_csv(f'../data/profile/{data_name}_profile.csv').iloc[:, 1:]
    if data_name == 'gas_price':
        days = [calendar.monthrange(2021, i+1)[1] for i in range(0,12)]
        random_matrix = np.repeat(0.8 + 0.4 * np.random.rand(1), days[0]*24)
        for i in days[1:]:
            random_matrix = np.hstack([random_matrix, np.repeat(0.8 + 0.4 * np.random.rand(1), i*24)])
        random_matrix = random_matrix.reshape(8760, -1)
    else:
        random_matrix = 0.8 + 0.4 * np.random.rand(*data.shape) # random range: 0.8 ~ 1.2
    data_with_random_noise = data * random_matrix
    data_with_random_noise = pd.DataFrame(data_with_random_noise, columns=data.columns)
    data_with_random_noise.to_csv(f'./randomized_raw_profile/{data_name}_randomized_profile.csv')

randomized_raw('PV')
randomized_raw('Wind')
randomized_raw('electricity_price')
randomized_raw('gas_price')
randomized_raw('load')
randomized_raw('sink_m')
randomized_raw('sink_p')

def randomized_states(data_name, predict_window):
    pdate = pd.date_range(start='2021-01-01 00:00:00', end=datetime.datetime.strptime('2021-12-31 23:00:00', '%Y-%m-%d %H:%M:%S')+datetime.timedelta(hours=-predict_window), freq='H')
    data_with_random_noise = pd.read_csv(f'./randomized_raw_profile/{data_name}_randomized_profile.csv')
    for z_index, z in enumerate(data_with_random_noise.columns[1:]):
        data_with_random_noise_and_predict_number = pd.DataFrame(columns=[f'predict_{i+1}' for i in range(predict_window)], index=pdate)
        for i in range(len(data_with_random_noise_and_predict_number)):
            data_with_random_noise_and_predict_number.iloc[i, :] = data_with_random_noise.iloc[(i + j for j in range(predict_window+1)), z_index+1].values[1:]
        data_with_random_noise_and_predict_number.to_csv(f'./randomized_states_profile/{data_name}_{z}_randomized_{predict_window}_profile.csv')

print(' * Creating randomized_states_profiles:')
for i in tqdm([1, 3, 6, 12]):
    randomized_states('PV', i)
    randomized_states('Wind', i)
    randomized_states('electricity_price', i)
    randomized_states('gas_price', i)
    randomized_states('load', i)
    randomized_states('sink_m', i)
    randomized_states('sink_p', i)