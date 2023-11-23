import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # pycharm bug
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from LSTM_train import LSTM_train

from tqdm import tqdm
import datetime

import time

def LSTM_predict_states(model, scaler_train, test_data, train_window, device):

    def create_inout_sequences(input_data, train_window, predict_window):
        inout_seq = []
        L = len(input_data)
        for i in range(L-train_window-predict_window+1):
            seq = input_data[i:i+train_window]
            #print('seq: ', seq)
            label = input_data[i+train_window:i+train_window+predict_window]
            #print('label: ', label)
            inout_seq.append((seq, label))
        return inout_seq

    model.eval().to(device)

    test_data = scaler_train.transform(test_data.reshape(-1, 1))
    test_data = torch.FloatTensor(test_data).view(-1).to(device)
    test_inout_seq = create_inout_sequences(test_data, train_window, predict_window)

    predict_data = []

    print(' * predicting:')
    '''
    for i in tqdm(range(len(test_data) - train_window)):
        seq = torch.FloatTensor(test_data[i:train_window + i])
        seq = seq.to(device)
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_sizes[0]).to(device),
                                 torch.zeros(1, 1, model.hidden_layer_sizes[0]).to(device))

            prediction = model(seq)
            prediction = prediction.cpu()
            predict_data.append(prediction.numpy().tolist())
            # seq = torch.cat((seq[1:], prediction), dim=0)
    '''
    for seq, labels in tqdm(test_inout_seq):
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_sizes[0]).to(device),
                             torch.zeros(1, 1, model.hidden_layer_sizes[0]).to(device))
        y_test_pred = model(seq)
        prediction = y_test_pred.cpu()
        predict_data.append(prediction.detach().numpy().tolist())

    return predict_data

'''
def save_model(model, model_name, dir):
    dir_ = os.path.join(dir, f'{model_name}.pt')
    torch.save(model, dir_)
'''

def filter_extreme_sigma(data1, data2, sigma=3):
    mean = data1.mean()
    std = data1.std()
    max_range = mean + sigma * std
    min_range = mean - sigma * std
    return np.clip(data2, min_range, max_range)

def plot_loss_curve(loss_curve, data_name, data_usecols, predict_window, tt):
    plt.figure()
    plt.grid(True)
    plt.plot(range(len(loss_curve)), loss_curve)
    if tt == 'train':
        plt.title(f'{data_name}_{data_usecols}_predict_{predict_window}_train_loss_curve')
        plt.savefig(f'./LSTM_predict_states_figs/{data_name}_{data_usecols}_predict_{predict_window}_train_loss_curve.png')
    elif tt == 'test':
        plt.title(f'{data_name}_{data_usecols}_predict_{predict_window}_test_loss_curve')
        plt.savefig(f'./LSTM_predict_states_figs/{data_name}_{data_usecols}_predict_{predict_window}_test_loss_curve.png')
    plt.show()
    plt.close()
    time.sleep(0.01)

def main(data_name, data_usecols, train_window, predict_window, epochs, device):
    data_ = pd.read_csv(f'./LSTM_train_profile/{data_name}_train_profile.csv', usecols=[f'{data_usecols}']).values.astype(float)
    train_data = data_[:7200]
    test_data = data_[7200:]
    validation_data = pd.read_csv(f'../data/profile/{data_name}_profile.csv', usecols=[f'{data_usecols}']).values.astype(float)
    #train_data = pd.read_csv(f'./LSTM_train_profile/{data_name}_train_profile.csv', usecols=[f'{data_usecols}']).values.astype(float)
    #test_data = pd.read_csv(f'../data/profile/{data_name}_profile.csv', usecols=[f'{data_usecols}']).values.astype(float)
    train_data = filter_extreme_sigma(train_data, train_data)
    print('data_name: ', data_name, '\ndata_usecols: ', data_usecols, '\npredict_window: ', predict_window)
    model, scaler_train, train_loss_curve, test_loss_curve = LSTM_train(train_data, test_data, train_window, epochs, predict_window, device)
    plot_loss_curve(train_loss_curve, data_name, data_usecols, predict_window, 'train')
    plot_loss_curve(test_loss_curve, data_name, data_usecols, predict_window, 'test')
    predict_data = LSTM_predict_states(model, scaler_train, validation_data, train_window, device)
    if data_name =='Electricity_price':
        scaler_predict = MinMaxScaler(feature_range=(0, scaler_train.data_max_))
    else:
        scaler_predict = MinMaxScaler(feature_range=(scaler_train.data_min_, scaler_train.data_max_))
    #predict_data = filter_extreme_sigma(np.array(predict_data)[:train_window-1], np.array(predict_data)).tolist()
    predict_data = scaler_predict.fit_transform(np.array(predict_data).reshape(-1, 1))
    actual_predict = predict_data.reshape(-1, predict_window)

    if data_name == 'sink_m':
        actual_predict = actual_predict.clip(max=0)
    else:
        actual_predict = actual_predict.clip(min=0)

    pdate = pd.date_range(start=datetime.datetime.strptime('2021-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')+datetime.timedelta(hours=train_window-1), end=datetime.datetime.strptime('2021-12-31 23:00:00', '%Y-%m-%d %H:%M:%S')+datetime.timedelta(hours=-predict_window), freq='H')
    df_predict = pd.DataFrame(actual_predict, columns=[f'predict_{i+1}' for i in range(predict_window)], index=pdate)

    if data_name == 'PV':
        for index_i, i  in enumerate(df_predict.index.hour):
            for index_j, j in enumerate(range(len(df_predict.columns))):
                if i+j+1 not in [7,8,9,10,11,12,13,14,15,16,17,18,19]:
                    df_predict.iloc[index_i, index_j] = 0

    df_predict.to_csv(f'./LSTM_predict_states_profile/{data_name}_{data_usecols}_predict_{predict_window}_profile.csv')

    plt.figure()
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(validation_data[train_window:train_window+1+240])
    plt.plot(df_predict.iloc[:1+240, 0].reset_index(drop=True))
    plt.title(f'{data_name}_{data_usecols}_predict_{predict_window}')
    plt.legend(['ture_data', 'predict_data'], loc='upper right')
    plt.savefig(f'./LSTM_predict_states_figs/{data_name}_{data_usecols}_predict_{predict_window}.png')
    plt.show()
    plt.close()
    print('----------------------------------------------------------')

if __name__ == '__main__':
    print('----------------------------------------------------------')
    train_window = 24
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device, '\n----------------------------------------------------------')

    for predict_window in [6, 3, 1]:
        main('PV', 'electricity', train_window, predict_window, epochs, device)
        main('Wind', 'electricity', train_window, predict_window, epochs, device)
        main('Electricity_price', 'buy_price', train_window, predict_window, epochs, device)
        main('Electricity_price', 'sell_price', train_window, predict_window, epochs, device)
        for i in [f'load{j}'for j in [1,4,5,7,9,10,11,12,14,15,16,18,19,23,24,26,28,30,31,32]]:
            main('load', i, train_window, predict_window, epochs, device)
        for i in [f'sink_m{j}' for j in [1,4,7,8,10,11,13,14,16,17,20,21,23,24,26,27,29,30,31,32]]:
            main('sink_m', i, train_window, predict_window, epochs, device)
        '''
        for i in [f'sink_p{j}' for j in [1,4,7,8,10,11,13,14,16,17,2,21,23,24,26,27,29,30,31,32]]:
            main('sink_p', i, train_window, predict_window, epochs, device)
        '''
