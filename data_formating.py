
import numpy as np
import pandas as pd 

def reshape_and_clean_data(df, valid_obs_number=None):

    df_counts = df.groupby(["PERMNO"]).count()

    if valid_obs_number is None:
        valid_obs_number = int(df_counts["PRC"].mode())
        print(f"valid_obs_number is set to {valid_obs_number}")

    df_valid = df_counts[df_counts["date"] == valid_obs_number]
    df_valid = df_valid[df_valid["PRC"] == valid_obs_number]

    print(f"Number of valid PERMNOs: {len(df_valid)}")
    clean_data = df[df["PERMNO"].isin(df_valid.index)]
    print(f"shape of df: {clean_data.shape}")

    temp = clean_data[(clean_data["FACSHR"] > 0) | (clean_data["FACSHR"] < 0)]
    permnos_with_splits = temp["PERMNO"].unique()
    print(f"found: {len(permnos_with_splits)} permnos with splits")

    print(f"removing stocks with splits...")
    clean_data = clean_data[~clean_data["PERMNO"].isin(permnos_with_splits)]
    print(f"shape of df: {clean_data.shape}")

    temp = clean_data[clean_data["PRC"] < 0]
    permnos_with_neg_prices = temp["PERMNO"].unique()
    print(f"found: {len(permnos_with_neg_prices)} permnos with negative prices")

    print(f"removing stocks with negative prices")
    clean_data = clean_data[~clean_data["PERMNO"].isin(permnos_with_neg_prices)]
    print(f"shape of df: {clean_data.shape}")

    pivoted = clean_data.pivot(index="date", columns="PERMNO", values="PRC")
    pivoted = pivoted.reset_index()

    return pivoted, valid_obs_number


def prepare_train_and_test(clean_data_first, clean_data_second, returns_period, n_train, n_validation, rows_to_keep):
    sampled_df = clean_data_first[clean_data_first.index.isin(rows_to_keep)]

    percent_df = sampled_df.pct_change(1).drop(sampled_df.index[[0]])
    percent_df_first = percent_df.drop(["date"], axis=1)

    percent_df_second = clean_data_second.pct_change(returns_period).drop(clean_data_second.index[range(0, returns_period)])
    percent_df_second = percent_df_second.drop(["date"], axis=1)

    common_permnos = np.intersect1d(percent_df_first.columns, percent_df_second.columns)

    print(f"common_permnos found: {len(common_permnos)}")

    percent_df_first = percent_df_first[common_permnos]
    percent_df_second = percent_df_second[common_permnos]

    assert(np.sum(percent_df_first.columns == percent_df_second.columns) == len(percent_df_first.columns))

    # stocks = percent_df_first = np.array(percent_df_first).T
    # labels = np.array(percent_df_second.iloc[[0]])[0]

    # all_ids = np.arange(len(stocks))

    all_permnos = percent_df_first.columns

    training_and_validation_permnos = np.random.choice(all_permnos, n_train + n_validation, replace=False)
    training_permnos = training_and_validation_permnos[:n_train]
    validation_permnos = training_and_validation_permnos[n_train:]
    print(f"using {len(training_permnos)} training ids")
    print(f"using {len(validation_permnos)} validation_ids")
    test_permnos = np.setdiff1d(all_permnos, training_and_validation_permnos)
    print(f"using {len(test_permnos)} test ids")

    training_stocks_df = percent_df_first[training_permnos]
    training_returns_df = percent_df_second[training_permnos]

    validation_stocks_df = percent_df_first[validation_permnos]
    validation_returns_df = percent_df_second[validation_permnos]

    test_stocks_df = percent_df_first[test_permnos]
    test_returns_df = percent_df_second[test_permnos]

    return (training_stocks_df, 
            training_returns_df,
            validation_stocks_df,
            validation_returns_df,
            test_stocks_df,
            test_returns_df)
