import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import time

def LSTM_train(train_data, train_window, epochs, predict_window):

    scaler_train = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler_train.fit_transform(train_data.reshape(-1, 1))
    train_data = torch.FloatTensor(train_data).view(-1)

    def create_inout_sequences(input_data, train_window):
        inout_seq = []
        L = len(input_data)
        for i in range(L-train_window-predict_window):
            train_seq = input_data[i:i+train_window]
            train_label = input_data[i+train_window:i+train_window+predict_window]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    train_inout_seq = create_inout_sequences(train_data, train_window)

    class LSTM(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=256, output_size=predict_window):
            super().__init__()
            self.hidden_layer_size = hidden_layer_size

            self.lstm = nn.LSTM(input_size, hidden_layer_size)
            self.linear = nn.Linear(hidden_layer_size, output_size)

        def forward(self, input_seq):
            lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
            predictions = self.linear(lstm_out.view(len(input_seq), -1))
            return predictions[-predict_window]

    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #print(model)

    print(' * training:')
    for i in tqdm(range(epochs)):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        #print(f'epoch: {i:3}')
        '''
        if i % 25 == 0:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        '''
    #rint(f'epoch: {i:3}, loss: {single_loss.item():10.10f}')
    time.sleep(0.01)

    return model, scaler_train