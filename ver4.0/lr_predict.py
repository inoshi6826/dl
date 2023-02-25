import os
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable

for year in range(2022, 2023):
  dir = '/content/drive/MyDrive/datasets/finance/exchange_rate/histdata/new_1h/'
  file_name = f'DAT_MT_USDJPY_1H_{year}.csv'
  path = dir + file_name

  df = pd.read_csv(path, index_col=0, parse_dates=True, usecols=lambda column: column!='Unnamed: 0' and column!='flag')
  df_flg = pd.read_csv(path, index_col=0, parse_dates=True, usecols=['datetime', 'flag'])
  x_data = torch.tensor(np.array(df), dtype=torch.float32)
  y_data = torch.tensor(np.array(df_flg), dtype=torch.float32)
  x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)
  print(x_train, x_test, y_train, y_test)

  train = TensorDataset(x_train, y_train)
  print(train[0])
  train_loader = DataLoader(train, batch_size=10, shuffle=True)

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        return x

input_dim = 8
output_dim = 1
model = LogisticRegressionModel(input_dim, output_dim)

criterion = torch.nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

loss_history = []
for epoch in range(100):
    total_loss = 0
    for x_train, y_train in train_loader:
        x_train = Variable(x_train)
        y_train = Variable(y_train)
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_history.append(total_loss)
    if (epoch +1) % 10 == 0:
        print(epoch + 1, total_loss)


plt.plot(loss_history)