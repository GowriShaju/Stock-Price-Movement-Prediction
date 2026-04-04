import pandas as pd

# For regression
def add_magnitude_target(df, horizon = 10):
    df["target_magnitude"] = df["Close"].shift(-horizon) / df["Close"] - 1
    return df


# For classification
def add_direction_target(df, horizon = 10):
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    df["target_direction"] = (future_return > 0).astype(int)
    return df


# Cleanup
def cleanup(df):
    before = len(df)
    df.dropna(inplace = True)
    after = len(df)

    print(f"Dropped {before - after} rows due to NaNs")
    return df


# SAVE
def save_data(df):
    df.to_csv("1_data/aapl_final.csv")
    print("Final dataset saved")
    return df


# MAIN PIPELINE
def create_targets(df):
    df = add_magnitude_target(df, horizon = 10)
    df = add_direction_target(df, horizon = 10) 

    df = cleanup(df)
    df = save_data(df)

    return df


if __name__ == "__main__":
    data = pd.read_csv(
        "1_data/aapl_with_indicators.csv",
        index_col = "Date",
        parse_dates = True
    )

    create_targets(data)

print(data["target_direction"].value_counts(normalize = True))
