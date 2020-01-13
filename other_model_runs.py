
import numpy as np 
import pandas as pd 

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from data_formating import reshape_and_clean_data, prepare_train_and_test

# data_2017 = pd.read_csv("2017_data.csv")
# data_2018 = pd.read_csv("2018_data.csv")

data_first = pd.read_csv("2017_data.csv")
data_second = pd.read_csv("2018_data.csv")

clean_data_first, _ = reshape_and_clean_data(data_first)
clean_data_second, _ = reshape_and_clean_data(data_second)

temp = prepare_train_and_test(clean_data_first,
                              clean_data_second,
                                returns_period=125,
                                n_train=4000,
                                n_validation=0,
                                rows_to_keep=range(0, 251, 20))


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

# model = RandomForestRegressor(n_estimators=1000)                              
model = LinearRegression()
model.fit(features, training_labels)

pred = model.predict(test_features)

mse_result = mean_squared_error(test_labels, pred)
print(f"mse_result: {mse_result}")

test = [[0.1] * 12,
        [0.2, -0.1] * 6,
        [0.05] * 12,
        [0.0] * 12,
        [-0.05] * 12,
        [-0.1] * 12,
]

special_test = pd.DataFrame({"mean": np.mean(test, axis=1),
                             "stdev": np.std(test, axis=1)
})

special_test_pred = model.predict(special_test)

for ele in special_test_pred:
    print(ele)

