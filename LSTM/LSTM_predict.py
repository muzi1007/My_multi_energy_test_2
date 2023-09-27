import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt

from LSTM_train import LSTM_train

def predict(model, test_data, train_window):
    model.eval()

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
    data = pd.read_csv('../data/profile/PV_profile.csv')
    data_input = data['electricity'].values.astype(float)
    data_input = data_input[:-180 * 24]

    test_data_size = 31 * 24
    train_data = data_input[:-test_data_size]
    test_data = data_input[-test_data_size:]

    train_window = 24
    epochs = 50
    model, scaler = LSTM_train(train_data, train_window, epochs)

    predict_data = predict(model, test_data, train_window)
    actual_predict = scaler.inverse_transform(np.array(predict_data[train_window:]).reshape(-1, 1))
    #print(actual_predict)

    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(test_data)
    plt.plot(actual_predict)
    plt.show()

'''
    model_name = 'Wind'
    dir = 'models'
    save_model(model, model_name, dir)
'''
