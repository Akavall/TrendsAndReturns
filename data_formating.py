
import numpy as np
import pandas as pd 

def reshape_and_clean_data(df, valid_obs_number):

    df_counts = df.groupby(["PERMNO"]).count()
    df_valid = df_counts[df_counts["date"] == valid_obs_number]
    df_valid = df_valid[df_valid["PRC"] == valid_obs_number]

    print(f"Number of valid PERMNOs: {len(df_valid)}")
    clean_data = df[df["PERMNO"].isin(df_valid.index)]
    print(f"shape of df: {clean_data.shape}")

    temp = clean_data[clean_data["FACSHR"] > 0]
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

    return pivoted


def prepare_train_and_test(clean_data_first, clean_data_second, returns_period, n_train, n_validation, rows_to_keep):
    sampled_df = clean_data_first[clean_data_first.index.isin(rows_to_keep)]
    percent_df = sampled_df.pct_change(1).drop(sampled_df.index[[0]])
    percent_df_first = percent_df.drop(["date"], axis=1)

    percent_df_second = clean_data_second.pct_change(returns_period).drop(clean_data_second.index[range(0, returns_period)])
    percent_df_second = percent_df_second.drop(["date"], axis=1)

    common_permnos = temp = np.intersect1d(percent_df_first.columns, percent_df_second.columns)

    print(f"common_permnos found: {len(common_permnos)}")

    percent_df_first = percent_df_first[common_permnos]
    percent_df_second = percent_df_second[common_permnos]

    assert(np.sum(percent_df_first.columns == percent_df_second.columns) == len(percent_df_first.columns))

    stocks = percent_df_first = np.array(percent_df_first).T
    labels = np.array(percent_df_second.iloc[[0]])[0]

    all_ids = np.arange(len(stocks))

    training_and_validation_ids = np.random.choice(np.arange(len(stocks)), n_train + n_validation, replace=False)
    training_ids = training_and_validation_ids[:n_train]
    validation_ids = training_and_validation_ids[n_train:]
    print(f"using {len(training_ids)} training ids")
    print(f"using {len(validation_ids)} validation_ids")
    test_ids = np.setdiff1d(all_ids, training_and_validation_ids)
    print(f"using {len(test_ids)} test ids")

    training_stocks = stocks[training_ids]
    training_labels = labels[training_ids]

    validation_stocks = stocks[validation_ids]
    validation_labels = labels[validation_ids]

    test_stocks = stocks[test_ids]
    test_labels = labels[test_ids]

    return training_stocks, training_labels, validation_stocks, validation_labels, test_stocks, test_labels