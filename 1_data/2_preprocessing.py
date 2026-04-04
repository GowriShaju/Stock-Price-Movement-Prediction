import pandas as pd
import numpy as np

def checking_columns(df):
    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        print(f"Mssing required column, {missing}")
    else:
        print("All required columns exist")
        
    print()
    print("--------------------------------------------------------------------------------------------------------")
    print()
    return df


def handling_null(df):
    print("Mssing before: /n", df.isnull().sum())

    price_cols = ["Open", "High", "Low", "Close"]
    df[price_cols] = df[price_cols].ffill()
    df['Volume'] = df['Volume'].fillna(0)
    df.dropna(inplace = True)

    print("Missing after: /n", df.isnull().sum())
    print()
    print("--------------------------------------------------------------------------------------------------------")
    print()
    return df


def remove_outliers(df):
    print("length before: ", len(df))

    from scipy import stats
    z_score = np.abs(stats.zscore(df['Close']))
    df = df[z_score <= 4]

    print("Length after: ", len(df))
    print()
    print("--------------------------------------------------------------------------------------------------------")
    print()
    return df


def save_preprocessed(df):
    df.to_csv("1_data/aapl_preprocessed.csv")
    return df


def report(df):
    print(f"  Shape         : {df.shape}")
    print(f"  Date range    : {df.index[0]} → {df.index[-1]}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Columns       : {list(df.columns)}")
    print("\n First 5 rows:")
    print(df.head())
    print("\n  Last 5 rows:")
    print(df.tail())
    print("\n  Return statistics:")
    print(df.describe())
    print()
    print("--------------------------------------------------------------------------------------------------------")
    print()


def preprocessing(df):
    df = checking_columns(df)
    df = handling_null(df)
    df = remove_outliers(df)
    df = save_preprocessed(df)
    report(df)
    return df


data = pd.read_csv("1_data/aapl.csv", index_col="Date", parse_dates=True)
preprocessing(data)
