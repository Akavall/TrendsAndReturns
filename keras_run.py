
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras import optimizers

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

from data_formating import reshape_and_clean_data, prepare_train_and_test

data_2017 = pd.read_csv("2017_data.csv")
data_2018 = pd.read_csv("2018_data.csv")

clean_data_2017 = reshape_and_clean_data(data_2017, valid_obs_number=251)
clean_data_2018 = reshape_and_clean_data(data_2018, valid_obs_number=251)

training_stocks, training_labels, test_stocks, test_labels = prepare_train_and_test(clean_data_2017,
                                                                                    clean_data_2018,
                                                                                    returns_period=125,
                                                                                    n_train=4000,
                                                                                    rows_to_keep=range(0, 251, 20))


training_stocks = np.expand_dims(training_stocks, axis=2)
test_stocks = np.expand_dims(test_stocks, axis=2)

model = Sequential()

# model.add(LSTM(1))
model.add(GRU(6, batch_input_shape=(32, 12, 1)))
model.add(Dense(1))

adam = optimizers.Adam(lr=0.001, decay=0.00001)


model.compile(loss="mean_squared_error", optimizer=adam)

history = model.fit(training_stocks, training_labels, epochs=100, validation_split=0.2)

# number of predictions needs to be divisible by batch size, hense the weirdness....

# https://stackoverflow.com/questions/43702481/why-does-keras-lstm-batch-size-used-for-prediction-have-to-be-the-same-as-fittin
# def create_model(batch_size, sl):
#     model = Sequential()
#     model.add(LSTM(1, batch_input_shape=(batch_size, sl, 1), stateful=True))
#     model.add(Dense(1))
#     return model

# import ipdb 
# ipdb.set_trace()

# model_predict = create_model(batch_size=5, sl=12)
# weights = model.get_weights()
# model_predict.set_weights(weights)


results = model.predict(test_stocks[:1024 + 32])[:, 0]
test_labels = test_labels[:1024 + 32]

mse_score = mean_squared_error(test_labels, results)

print(f"mse_score: {mse_score}")

another_test = np.zeros((32, 12))

test = [[0.1] * 12,
        [0.2, -0.1] * 6,
        [0.05] * 12,
        [0.0] * 12,
        [-0.05] * 12,
        [-0.1] * 12,]

for i in range(len(test)):
    another_test[i] = test[i]

another_test = np.expand_dims(another_test, axis=2)

results = model.predict(another_test)

print(results[:6])