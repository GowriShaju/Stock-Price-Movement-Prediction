import pandas as pd

# ADD FUTURE RETURN
def add_magnitude_target(df, horizon = 10):
    df["target_magnitude"] = df["Close"].shift(-horizon) / df["Close"] - 1
    print(f"{horizon}-day future return added")
    return df


# CREATE DIRECTION TARGET
def add_direction_target(df, threshold = 0.02):
    df["target_direction"] = (df["Close"] > df["Close"].rolling(20).mean()).astype(int)

    up = (df["target_direction"] == 1).sum()
    down = (df["target_direction"] == 0).sum()

    print(f"Target created → UP: {up}, DOWN: {down}")
    return df


# CLEANUP
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
    df = add_direction_target(df, threshold = 0.02)

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
print(data["target_direction"].value_counts(normalize = True))
