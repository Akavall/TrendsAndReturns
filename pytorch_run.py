import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

import torch
from torch import nn
import torch.nn.functional as F

from pytorch_model import RNNRegressor

from data_formating import reshape_and_clean_data, prepare_train_and_test

import json

data_first = pd.read_csv("2017_data.csv")
data_second = pd.read_csv("2018_data.csv")

clean_data_first, valid_obs_first = reshape_and_clean_data(data_first)
clean_data_second, valid_obs_second = reshape_and_clean_data(data_second)

temp = prepare_train_and_test(clean_data_first,
                              clean_data_second,
                                returns_period=125,
                                n_train=3200,
                                n_validation=800,
                                rows_to_keep=range(0, valid_obs_first, 20))


def pytorch_run(temp):

    training_stocks_df, training_returns_df, validation_stocks_df, validation_returns_df, test_stocks_df, test_returns_df = temp

    training_stocks = training_stocks_df.T 
    training_labels = training_returns_df.iloc[[0]].T
    validation_stocks = validation_stocks_df.T
    validation_labels = validation_returns_df.iloc[[0]].T
    test_stocks = test_stocks_df.T 
    test_labels = test_returns_df.iloc[[0]].T


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

    for epoch in range(20):

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



    # test = [[0.1] * 12,
    #         [0.2, -0.1] * 6,
    #         [0.05] * 12,
    #         [0.0] * 12,
    #         [-0.05] * 12,
    #         [-0.1] * 12,
    # ]
    # test_stock = torch.from_numpy(np.array(test)).float()
    # test_stock = test_stock.T.reshape(test_stock.shape[1], test_stock.shape[0], -1)

    # with torch.no_grad():
    #     result = model(test_stock)

    # print(result)

    with torch.no_grad():

        test_stocks_torch = test_stocks_torch.T.reshape(test_stocks_torch.shape[1], test_stocks_torch.shape[0], -1)

        preds_torch = model(test_stocks_torch)
        preds = preds_torch.cpu().detach().numpy()[0]

        mse_score = mean_squared_error(test_labels, preds)
        print(f"mse_score: {mse_score}")
        # predicting that price didn't change
        mse_score_zeros = mean_squared_error(test_labels, np.zeros(len(preds)))
        print(f"mse_score zeros: {mse_score_zeros}")

        if mse_score < mse_score_zeros:
            print(f"Success: did better that zeros by {mse_score_zeros - mse_score}")
        else: 
            print(f"Could not beat zeros, missed by: {mse_score_zeros - mse_score}")

        return mse_score


# Let's do an ugly hack for OLS

def other_model_run(temp):

    training_stocks_df, training_returns_df, _, _, test_stocks_df, test_returns_df = temp

    training_stocks = training_stocks_df.T 
    training_labels = training_returns_df.iloc[[0]].T
    test_stocks = test_stocks_df.T 
    test_labels = test_returns_df.iloc[[0]].T

    training_mean = np.mean(training_stocks, axis=1)
    training_stdev = np.std(training_stocks, axis=1)

    features = pd.DataFrame({"mean": training_mean, 
                            "stdev": training_stdev})

    test_features = pd.DataFrame({"mean": np.mean(test_stocks, axis=1),
                                "stdev": np.std(test_stocks, axis=1)})
                                
    model = LinearRegression()
    model.fit(features, training_labels)

    other_model_preds = model.predict(test_features)

    other_model_mse_score = mean_squared_error(test_labels, other_model_preds)
    print(f"other_model_mse_result: {other_model_mse_score}")

    return other_model_mse_score 

if __name__ == "__main__":

    data_first = pd.read_csv("2017_data.csv")
    data_second = pd.read_csv("2018_data.csv")

    clean_data_first, valid_obs_first = reshape_and_clean_data(data_first)
    clean_data_second, valid_obs_second = reshape_and_clean_data(data_second)

    pytorch_result = []
    other_model_result = []

    for i in range(2):

        temp = prepare_train_and_test(clean_data_first,
                                    clean_data_second,
                                        returns_period=125,
                                        n_train=3200,
                                        n_validation=800,
                                        rows_to_keep=range(0, valid_obs_first, 20))

        pytorch_mse = pytorch_run(temp)
        pytorch_result.append(pytorch_mse)

        other_model_mse = other_model_run(temp)
        other_model_result.append(other_model_mse)


    all_result = {"pytorch": pytorch_result, "other_model": other_model_result}

    with open("test_result.json", "w") as f:
        json.dump(all_result, f)






    
