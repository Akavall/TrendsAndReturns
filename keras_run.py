
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras import optimizers

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

from data_formating import reshape_and_clean_data, prepare_train_and_test

data_first = pd.read_csv("2016_data.csv")
data_second = pd.read_csv("2017_data.csv")

clean_data_first, valid_obs_first = reshape_and_clean_data(data_first)
clean_data_second, valid_obs_second = reshape_and_clean_data(data_second)

# training_stocks, training_labels, _, _, test_stocks, test_labels = prepare_train_and_test(clean_data_first,
#                                                                                     clean_data_second,
#                                                                                     returns_period=125,
#                                                                                     n_train=4000,
#                                                                                     n_validation=0,
#                                                                                     rows_to_keep=range(0, valid_obs_second, 20))


temp = prepare_train_and_test(clean_data_first,
                              clean_data_second,
                                returns_period=125,
                                n_train=4000,
                                n_validation=0,
                                rows_to_keep=range(0, valid_obs_second, 20))


training_stocks_df, training_returns_df, _, _, test_stocks_df, test_returns_df = temp

training_stocks = training_stocks_df.T 
training_labels = training_returns_df.iloc[[0]].T
test_stocks = test_stocks_df.T 
test_labels = test_returns_df.iloc[[0]].T


training_stocks = np.expand_dims(training_stocks, axis=2)
test_stocks = np.expand_dims(test_stocks, axis=2)

model = Sequential()

model.add(GRU(6))
model.add(Dense(1))

adam = optimizers.Adam(lr=0.001, decay=0.00001)


model.compile(loss="mean_squared_error", optimizer=adam)

history = model.fit(training_stocks, np.array(training_labels), epochs=10, validation_split=0.2, batch_size=32)

results = model.predict(test_stocks)[:, 0]
test_labels = test_labels

mse_score = mean_squared_error(test_labels, results)

print(f"mse_score: {mse_score}")

test = [[0.1] * 12,
        [0.2, -0.1] * 6,
        [0.05] * 12,
        [0.0] * 12,
        [-0.05] * 12,
        [-0.1] * 12,]

another_test = np.expand_dims(np.array(test), axis=2)

results = model.predict(another_test)

print(results)