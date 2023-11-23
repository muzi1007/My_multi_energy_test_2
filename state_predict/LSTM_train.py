import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import time

from torch.optim.lr_scheduler import StepLR

def LSTM_train(train_data, test_data, train_window, epochs, predict_window, device, patience=20):

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

    scaler_train = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler_train.fit_transform(train_data.reshape(-1, 1))
    train_data = torch.FloatTensor(train_data).view(-1).to(device)
    train_inout_seq = create_inout_sequences(train_data, train_window, predict_window)

    if test_data is not None:
        test_data = torch.FloatTensor(test_data).view(-1).to(device)
        test_inout_seq = create_inout_sequences(test_data, train_window, predict_window)

    class LSTM(nn.Module):
        def __init__(self, input_size=1, hidden_layer_sizes=[32, 16, 8], output_size=predict_window):
            super().__init__()
            self.hidden_layer_sizes = hidden_layer_sizes

            self.lstm1 = nn.LSTM(input_size, hidden_layer_sizes[0])
            self.lstm2 = nn.LSTM(hidden_layer_sizes[0], hidden_layer_sizes[1])
            self.lstm3 = nn.LSTM(hidden_layer_sizes[1], hidden_layer_sizes[2])
            self.linear = nn.Linear(hidden_layer_sizes[2], output_size)

        def forward(self, input_seq):
            lstm_out1, _ = self.lstm1(input_seq.view(len(input_seq), 1, -1))
            lstm_out2, _ = self.lstm2(lstm_out1)
            lstm_out3, _ = self.lstm3(lstm_out2)
            predictions = self.linear(lstm_out3.view(len(input_seq), -1))
            return predictions[-predict_window]

    model = LSTM(hidden_layer_sizes=[32, 16, 8]).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
    #print(model)

    #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # 每 step_size 個 epoch 衰減學習率為原來的 gamma

    print(' * training:')
    train_loss_curve = []
    test_loss_curve = []
    best_test_loss = float('inf')
    early_stop_counter = 0

    for i in tqdm(range(epochs)):
        model.train()
        train_losses = []
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_sizes[0]).to(device),
                                 torch.zeros(1, 1, model.hidden_layer_sizes[0]).to(device))

            y_train_pred = model(seq)
            #print('\ny_train_pred: ', y_train_pred, 'labels: ', labels)

            train_loss = loss_function(y_train_pred, labels)
            train_loss.backward()
            train_losses.append(train_loss.cpu().detach().numpy())

            optimizer.step()

        avg_train_loss = np.mean(train_losses)
        train_loss_curve.append(avg_train_loss.item())

        #scheduler.step()  # 每一個 epoch 後更新學習率

        if test_data is not None:
            model.eval()
            with torch.no_grad():
                test_losses = []
                for seq, labels in test_inout_seq:
                    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_sizes[0]).to(device),
                                         torch.zeros(1, 1, model.hidden_layer_sizes[0]).to(device))

                    y_test_pred = model(seq)

                    test_loss = loss_function(y_test_pred, labels)
                    test_losses.append(test_loss.cpu().detach().numpy())
                avg_test_loss = np.mean(test_losses)
                test_loss_curve.append(avg_test_loss.item())

                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    early_stop_counter = 0
                    best_model = model.state_dict()

                else:
                    early_stop_counter += 1
                    model.load_state_dict(best_model)

                if early_stop_counter >= patience:
                    print(f' - Stop early. Test loss for consecutive {patience} epochs without improvement.')
                    model.load_state_dict(best_model)
                    print(f' - The best model on the test set have been loaded.')
                    return model, scaler_train, train_loss_curve, test_loss_curve

    model.load_state_dict(best_model)
    print(f' - Done all epochs, the best model on the test set have been loaded.')
    time.sleep(0.01)
    return model, scaler_train, train_loss_curve, test_loss_curve