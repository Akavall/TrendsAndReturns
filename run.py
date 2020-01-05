import numpy as np
import pandas as pd 

from sklearn.metrics import mean_squared_error

import torch
from torch import nn
import torch.nn.functional as F

from model import RNNRegressor

from data_formating import reshape_and_clean_data, prepare_train_and_test

data_2017 = pd.read_csv("2017_data.csv")
data_2018 = pd.read_csv("2018_data.csv")

clean_data_2017 = reshape_and_clean_data(data_2017, valid_obs_number=251)
clean_data_2018 = reshape_and_clean_data(data_2018, valid_obs_number=251)

training_stocks, training_labels, validation_stocks, validation_labels, test_stocks, test_labels = prepare_train_and_test(clean_data_2017,
                                                                                    clean_data_2018,
                                                                                    returns_period=125,
                                                                                    n_train=3200,
                                                                                    n_validation=800,
                                                                                    rows_to_keep=range(0, 251, 20)
                                                                                    )


#convert to torch types
stocks_torch = torch.from_numpy(np.array(training_stocks)).float()
stocks_torch = stocks_torch.T.reshape(stocks_torch.shape[1], stocks_torch.shape[0], -1)                                                                                  
validation_stocks_torch = torch.from_numpy(np.array(validation_stocks)).float()
validation_stocks_torch = validation_stocks_torch.T.reshape(validation_stocks_torch.shape[1],
                                                            validation_stocks_torch.shape[0],
                                                            -1
)

labels_torch = torch.from_numpy(np.array(training_labels)).float()

test_stocks_torch = torch.from_numpy(np.array(test_stocks)).float()

input_size = 1
hidden_size = 16
output_size = 1

model = RNNRegressor(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

BATCH_SIZE = 32

for epoch in range(100):

    permutation = torch.randperm(stocks_torch.size(1))
    epoch_loss = 0

    for i in range(0, stocks_torch.size(1), BATCH_SIZE):

        optimizer.zero_grad()

        indicies = permutation[i: i+BATCH_SIZE]

        batch_stocks = stocks_torch[:, indicies, :]
        batch_labels = labels_torch[indicies]
        batch_labels = batch_labels.reshape(1, -1)

        output = model(batch_stocks)

        loss = criterion(output, batch_labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss

    with torch.no_grad():

        preds_torch = model(validation_stocks_torch)
        preds = preds_torch.cpu().detach().numpy()[0]

        mse_score = mean_squared_error(validation_labels, preds)

    print(f"Epoch: {epoch}, loss: {epoch_loss}, validation mse_score: {mse_score}")



test = [[0.1] * 12,
        [0.2, -0.1] * 6,
        [0.05] * 12,
        [0.0] * 12,
        [-0.05] * 12,
        [-0.1] * 12,
]
test_stock = torch.from_numpy(np.array(test)).float()
test_stock = test_stock.T.reshape(test_stock.shape[1], test_stock.shape[0], -1)

with torch.no_grad():
    result = model(test_stock)

print(result)

with torch.no_grad():

    test_stocks_torch = test_stocks_torch.T.reshape(test_stocks_torch.shape[1], test_stocks_torch.shape[0], -1)

    preds_torch = model(test_stocks_torch)
    preds = preds_torch.cpu().detach().numpy()[0]

    mse_score = mean_squared_error(test_labels, preds)

    print(f"mse_score: {mse_score}")

    
