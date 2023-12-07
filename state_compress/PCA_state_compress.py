import numpy as np
from sklearn.decomposition import PCA

import pandas as pd
from tqdm import tqdm

def PCA_LSTM_predict_states(data_name, data_numbers, predict_window):
    if data_name in ['load', 'sink_m']:
        states = []
        for i_index, i in tqdm(enumerate(data_numbers)):
            # load states
            predicts = pd.read_csv(f'../state_predict/LSTM_predict_states_profile/{data_name}_{data_name}{i}_predict_{predict_window}_profile.csv', usecols=[f'predict_{k+1}' for k in range(0, predict_window)])[:8736].astype(float).values
            # normalize states
            mean = np.mean(predicts, axis=0)
            std = np.std(predicts, axis=0)
            normalized_predict = (predicts - mean) / std

            # PCA
            if i_index == 0:
                pca_ = PCA()
                pca_.fit(normalized_predict)

            n_components = 1

            reduced_predicts = pca_.transform(normalized_predict)[:, :n_components]
            states.append(reduced_predicts)
        states = np.array(states).reshape(-1, len(predicts)).T

        # normalize states
        mean = np.mean(states, axis=0)
        std = np.std(states, axis=0)
        normalized_states = (states - mean) / std

        # PCA
        pca = PCA()
        pca.fit(normalized_states)

        n_components = 1

        reduced_states = pca.transform(normalized_states)[:, :n_components]
        df_reduced_states = pd.DataFrame(reduced_states, columns=['reduced_states'])
        df_reduced_states.to_csv(f'./LSTM_predict_{predict_window}_reduced_states/{data_name}_predict_{predict_window}_reduced_states.csv')
        print('----------------------------------------------------------')

    else:
        # load states
        predicts = pd.read_csv(f'../state_predict/LSTM_predict_states_profile/{data_name}_predict_{predict_window}_profile.csv', usecols=[f'predict_{k + 1}' for k in range(0, predict_window)])[:8736].astype(float).values
        # normalize states
        mean = np.mean(predicts, axis=0)
        std = np.std(predicts, axis=0)
        normalized_predict = (predicts - mean) / std

        # PCA
        pca = PCA()
        pca.fit(normalized_predict)

        n_components = 1

        reduced_states = pca.transform(normalized_predict)[:, :n_components]
        df_reduced_states = pd.DataFrame(reduced_states, columns=['reduced_states'])
        df_reduced_states.to_csv(f'./LSTM_predict_{predict_window}_reduced_states/{data_name}_randomized_{predict_window}_reduced_states.csv')

def PCA_randomized_states(data_name, data_numbers, predict_window):
    if data_name in ['load', 'sink_m']:
        states = []
        for i_index, i in tqdm(enumerate(data_numbers)):
            # load states
            predicts = pd.read_csv(f'../state_random_noise/randomized_states_profile/{data_name}_{data_name}{i}_randomized_{predict_window}_profile.csv', usecols=[f'predict_{k+1}' for k in range(0, predict_window)])[:8736].astype(float).values
            # normalize states
            mean = np.mean(predicts, axis=0)
            std = np.std(predicts, axis=0)
            normalized_predict = (predicts - mean) / std

            # PCA
            if i_index == 0:
                pca_ = PCA()
                pca_.fit(normalized_predict)

            n_components = 1

            reduced_predicts = pca_.transform(normalized_predict)[:, :n_components]
            states.append(reduced_predicts)
        states = np.array(states).reshape(-1, len(predicts)).T

        # normalize states
        mean = np.mean(states, axis=0)
        std = np.std(states, axis=0)
        normalized_states = (states - mean) / std

        # PCA
        pca = PCA()
        pca.fit(normalized_states)

        n_components = 1

        reduced_states = pca.transform(normalized_states)[:, :n_components]
        df_reduced_states = pd.DataFrame(reduced_states, columns=['reduced_states'])
        df_reduced_states.to_csv(f'./randomized_{predict_window}_reduced_states/{data_name}_randomized_{predict_window}_reduced_states.csv')
        print('----------------------------------------------------------')

    else:
        # load states
        predicts = pd.read_csv(f'../state_random_noise/randomized_states_profile/{data_name}_randomized_{predict_window}_profile.csv', usecols=[f'predict_{k + 1}' for k in range(0, predict_window)])[:8736].astype(float).values
        # normalize states
        mean = np.mean(predicts, axis=0)
        std = np.std(predicts, axis=0)
        normalized_predict = (predicts - mean) / std

        # PCA
        pca = PCA()
        pca.fit(normalized_predict)

        n_components = 1

        reduced_states = pca.transform(normalized_predict)[:, :n_components]
        df_reduced_states = pd.DataFrame(reduced_states, columns=['reduced_states'])
        df_reduced_states.to_csv(f'./randomized_{predict_window}_reduced_states/{data_name}_randomized_{predict_window}_reduced_states.csv')


if __name__ == '__main__':
    '''
    print(' * producing - LSTM_predict_states:')
    PCA_LSTM_predict_states('load',   [1,4,5,7,9,10,11,12,14,15,16,18,19,23,24,26,28,30,31,32],  1)
    PCA_LSTM_predict_states('sink_m', [1,4,7,8,10,11,13,14,16,17,20,21,23,24,26,27,29,30,31,32], 1)
    PCA_LSTM_predict_states('load',   [1,4,5,7,9,10,11,12,14,15,16,18,19,23,24,26,28,30,31,32],  3)
    PCA_LSTM_predict_states('sink_m', [1,4,7,8,10,11,13,14,16,17,20,21,23,24,26,27,29,30,31,32], 3)
    PCA_LSTM_predict_states('load',   [1,4,5,7,9,10,11,12,14,15,16,18,19,23,24,26,28,30,31,32],  6)
    PCA_LSTM_predict_states('sink_m', [1,4,7,8,10,11,13,14,16,17,20,21,23,24,26,27,29,30,31,32], 6)
    print(' * producing - randomized_states:')
    PCA_randomized_states('load',   [1,4,5,7,9,10,11,12,14,15,16,18,19,23,24,26,28,30,31,32],  1)
    PCA_randomized_states('sink_m', [1,4,7,8,10,11,13,14,16,17,20,21,23,24,26,27,29,30,31,32], 1)
    PCA_randomized_states('load',   [1,4,5,7,9,10,11,12,14,15,16,18,19,23,24,26,28,30,31,32],  3)
    PCA_randomized_states('sink_m', [1,4,7,8,10,11,13,14,16,17,20,21,23,24,26,27,29,30,31,32], 3)
    PCA_randomized_states('load',   [1,4,5,7,9,10,11,12,14,15,16,18,19,23,24,26,28,30,31,32],  6)
    PCA_randomized_states('sink_m', [1,4,7,8,10,11,13,14,16,17,20,21,23,24,26,27,29,30,31,32], 6)
    '''
    print(' * producing - LSTM_predict_states:')
    PCA_LSTM_predict_states('Electricity_price_sell_price', 0, 1)
    PCA_LSTM_predict_states('Electricity_price_sell_price', 0, 3)
    PCA_LSTM_predict_states('Electricity_price_sell_price', 0, 6)
    PCA_LSTM_predict_states('Electricity_price_buy_price', 0, 1)
    PCA_LSTM_predict_states('Electricity_price_buy_price', 0, 3)
    PCA_LSTM_predict_states('Electricity_price_buy_price', 0, 6)
    PCA_LSTM_predict_states('PV_electricity', 0, 1)
    PCA_LSTM_predict_states('PV_electricity', 0, 3)
    PCA_LSTM_predict_states('PV_electricity', 0, 6)
    PCA_LSTM_predict_states('Wind_electricity', 0, 1)
    PCA_LSTM_predict_states('Wind_electricity', 0, 3)
    PCA_LSTM_predict_states('Wind_electricity', 0, 6)
    print('done.')
    print(' * producing - randomized_states:')
    PCA_randomized_states('Electricity_price_sell_price', 0, 1)
    PCA_randomized_states('Electricity_price_sell_price', 0, 3)
    PCA_randomized_states('Electricity_price_sell_price', 0, 6)
    PCA_randomized_states('Electricity_price_buy_price', 0, 1)
    PCA_randomized_states('Electricity_price_buy_price', 0, 3)
    PCA_randomized_states('Electricity_price_buy_price', 0, 6)
    PCA_randomized_states('PV_electricity', 0, 1)
    PCA_randomized_states('PV_electricity', 0, 3)
    PCA_randomized_states('PV_electricity', 0, 6)
    PCA_randomized_states('Wind_electricity', 0, 1)
    PCA_randomized_states('Wind_electricity', 0, 3)
    PCA_randomized_states('Wind_electricity', 0, 6)
    print('done.')
