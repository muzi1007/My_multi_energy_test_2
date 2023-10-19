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
import calendar

def predict(model, scaler_train, test_data, train_window):
    model.eval()

    test_data = scaler_train.transform(test_data.reshape(-1, 1))
    test_data = torch.FloatTensor(test_data).view(-1)

    test_data = test_data.tolist()
    predict_data = []

    for i in tqdm(range(len(test_data) - train_window)):
        seq = torch.FloatTensor(test_data[i:train_window + i])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))

            prediction = model(seq)
            predict_data.append(prediction.numpy().tolist())
            #seq = torch.cat((seq[1:], prediction), dim=0)

    return predict_data

'''
def save_model(model, model_name, dir):
    dir_ = os.path.join(dir, f'{model_name}.pt')
    torch.save(model, dir_)
'''

def main(data_name, data_usecols, train_window, predict_window, epochs):
    train_data = pd.read_csv(f'./data/LSTM_train_data/{data_name}_train_profile.csv', usecols=[f'{data_usecols}']).values.astype(float)
    test_data = pd.read_csv(f'../data/profile/{data_name}_profile.csv', usecols=[f'{data_usecols}']).values.astype(float)
    model, scaler_train = LSTM_train(train_data, train_window, epochs, predict_window)
    predict_data = predict(model, scaler_train, test_data, train_window)
    scaler_predict = MinMaxScaler(feature_range=(scaler_train.data_min_, scaler_train.data_max_))
    predict_data = scaler_predict.fit_transform(np.array(predict_data).reshape(-1, 1))
    actual_predict = predict_data.reshape(-1, predict_window)

    if data_name == 'sink_m':
        actual_predict = actual_predict.clip(max=0)
    elif data_name =='Electricity_price':
        pass
    else:
        actual_predict = actual_predict.clip(min=0)

    pdate = pd.date_range(start='2021-01-02 00:00:00', end='2021-12-31 23:00:00', freq='H')
    df_predict = pd.DataFrame(actual_predict, columns=[f'predict_{i+1}' for i in range(predict_window)], index=pdate)

    if data_name == 'PV':
        for index_i, i  in enumerate(df_predict.index.hour):
            for index_j, j in enumerate(range(len(df_predict.columns))):
                if i+j+1 not in [7,8,9,10,11,12,13,14,15,16,17,18,19]:
                    df_predict.iloc[index_i, index_j] = 0
    df_predict.to_csv(f'./data/LSTM_predict_data/{data_name}_{data_usecols}_predict_{predict_window}_profile.csv')

    plt.figure()
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(test_data[24:240])
    plt.plot(actual_predict[:240-24,0])
    plt.title(f'{data_name}_{data_usecols}_predict_{predict_window}')
    plt.legend(['ture_data', 'predict_data'], loc='upper right')
    plt.savefig(f'./figs/{data_name}_{data_usecols}_predict_{predict_window}.png')
    plt.show()
    plt.close()


if __name__ == '__main__':

    train_window = 24
    epochs = 50

    for predict_window in [1, 3, 12]:
        main('PV', 'electricity', train_window, predict_window, epochs)
        main('Wind', 'electricity', train_window, predict_window, epochs)
        main('Electricity_price', 'buy_price', train_window, predict_window, epochs)
        main('Electricity_price', 'sell_price', train_window, predict_window, epochs)
        for i in [f'load{j}'for j in [1,4,5,7,9,10,11,12,14,15,16,18,17,23,24,26,28,30,31,32]]:
            main('load', i, train_window, predict_window, epochs)
        for i in [f'sink_m{j}' for j in [1,4,7,8,10,11,13,14,16,17,2,21,23,24,26,27,29,30,31,32]]:
            main('sink_m', i, train_window, predict_window, epochs)
        '''
        for i in [f'sink_p{j}' for j in [1,4,7,8,10,11,13,14,16,17,2,21,23,24,26,27,29,30,31,32]]:
            main('sink_p', i, train_window, predict_window, epochs)
        '''