import torch
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as mae
from LSTM_train import LSTM_train

def predict(model, test_data, train_window):
    model.eval()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    test_data = scaler.fit_transform(test_data.reshape(-1, 1))
    test_data = torch.FloatTensor(test_data).view(-1)

    test_data = test_data.tolist()
    predict_data = []

    for i in range(len(test_data)-train_window):
        seq = torch.FloatTensor(test_data[i:train_window+i])

        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))

            predict_data.append(model(seq).item())

    return predict_data

def save_model(model, model_name, dir):
    dir_ = os.path.join(dir, f'{model_name}.pt')
    torch.save(model, dir_)

if __name__ == '__main__':
    train_data = pd.read_csv('./data/LSTM_train_data/Wind_train_profile.csv', usecols=['electricity']).values.astype(float)
    train_data = pd.read_csv('./data/LSTM_train_data/sink_p_train_profile.csv', usecols=['sink_p1']).values.astype(float)
    #train_data = pd.read_csv('../data/profile/PV_profile.csv', usecols=['electricity']).values.astype(float)

    #test_data = pd.read_csv('../data/profile/Wind_profile.csv', usecols=['electricity']).values.astype(float)
    test_data = pd.read_csv('../data/profile/sink_p_profile.csv', usecols=['sink_p1']).values.astype(float)

    train_window = 24
    epochs = 1
    model, scaler = LSTM_train(train_data, train_window, epochs)
    predict_data = predict(model, test_data, train_window)
    actual_predict = scaler.inverse_transform(np.array(predict_data).reshape(-1, 1))
    #print(actual_predict)

    actual_predict = actual_predict.clip(min=0)
    #actual_predict = actual_predict.clip(max=0)

    predict_mae = mae(test_data[24:], actual_predict)
    print('predict_mae: ', predict_mae)

    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(test_data[24:])
    plt.plot(actual_predict)
    plt.show()

    pdate = pd.date_range(start='2021-01-02 00:00:00', end='2021-12-31 23:00:00', freq='H')
    df_predict = pd.DataFrame(actual_predict, columns=['predict'], index=pdate)

    df_predict.to_csv('./data/LSTM_predict_data/sink_p1_predict_profile.csv')
    