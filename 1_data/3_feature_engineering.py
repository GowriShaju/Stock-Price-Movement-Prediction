import numpy as np
import pandas as pd


# ===== CORE FEATURES =====

def add_returns(df):
    df["return_1"] = df["Close"].pct_change(1)
    df["return_2"] = df["Close"].pct_change(2)
    df["return_acc"] = df["return_1"].diff() 
    print("Returns added")
    return df


def add_momentum(df):
    df["momentum_3"] = df["Close"] / df["Close"].shift(3) - 1
    df["momentum_7"] = df["Close"] / df["Close"].shift(7) - 1
    df["momentum_14"] = df["Close"] / df["Close"].shift(14) - 1
    df["momentum_change"] = df["momentum_3"].diff() 
    return df

def smooth_features(df):
    for col in ["return_1", "return_2", "momentum_3", "momentum_7", "momentum_14"]:
        df[f"{col}_smooth"] = df[col].rolling(3).mean()
    return df

def add_long_trend(df):
    df["trend_20"] = df["Close"].pct_change(20)
    df["trend_30"] = df["Close"].pct_change(30)
    return df

def add_price_transforms(df):
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["oc_change"] = (df["Close"] - df["Open"]) / df["Open"]
    print("Price transforms added")
    return df


def transform_volume(df):
    df["Volume"] = np.log1p(df["Volume"])
    print("Volume transformed")
    return df

def add_ema_features(df):
    df["ema_10"] = df["Close"].ewm(span=10).mean()
    df["ema_20"] = df["Close"].ewm(span=20).mean()

    df["ema_diff"] = df["ema_10"] - df["ema_20"]
    df["price_vs_ema"] = df["Close"] - df["ema_10"]

    return df


def add_volatility_feature(df):
    df["volatility_3"] = df["Close"].pct_change().rolling(3).std() 
    df["volatility_10"] = df["Close"].pct_change().rolling(10).std()
    return df


def add_volume_features(df):
    df["vol_change"] = df["Volume"].pct_change()

    df["vol_spike"] = (
        df["vol_change"] > df["vol_change"].rolling(10).mean()
    ).astype(int)

    return df


def add_price_position(df):
    rolling_min = df["Close"].rolling(20).min()
    rolling_max = df["Close"].rolling(20).max(0)
    df["price_norm"] = df["Close"] / df["Close"].rolling(20).mean() 
    df["price_position"] = (df["Close"] - rolling_min) / (rolling_max - rolling_min)

    return df



# ===== INDICATORS =====

def add_rsi(df, period=14):
    diff = df['Close'].diff()
    gains = diff.clip(lower=0)
    losses = -diff.clip(upper=0)

    avg_gains = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_losses = losses.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gains / avg_losses
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()

    return df


def add_bollinger_bands(df, period=20, std_dev=2):
    df["BB_Middle"] = df["Close"].rolling(window=period).mean()
    rolling_std = df["Close"].rolling(window=period).std()

    df["BB_Upper"] = df["BB_Middle"] + (std_dev * rolling_std)
    df["BB_Lower"] = df["BB_Middle"] - (std_dev * rolling_std)

    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Pct"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    df.drop(["BB_Upper", "BB_Lower"], axis=1, inplace=True)

    return df


def add_volatility(df):
    df["volatility_10"] = df["Close"].pct_change().rolling(10).std()
    return df


def add_trend_feature(df):
    df["trend_5"] = df["Close"].pct_change(5)
    df["trend_10"] = df["Close"].pct_change(10)
    return df


# ===== CLEANUP =====

def final_cleanup(df):
    df.dropna(inplace=True)
    return df


# ===== MAIN PIPELINE =====

def feature_engineering(df):

    df = add_returns(df)
    df = add_momentum(df)
    df = smooth_features(df)
    df = add_long_trend(df)
    df = add_price_transforms(df)
    df = transform_volume(df)
    df = add_ema_features(df)
    df = add_volatility_feature(df)
    df = add_volume_features(df)
    df = add_price_position(df)

    df = add_rsi(df)
    df = add_macd(df) 
    df = add_bollinger_bands(df)
    df = add_volatility(df)
    df = add_trend_feature(df)

    df = final_cleanup(df)

    df.to_csv("1_data/aapl_with_indicators.csv")
    print("Final dataset ready")

    return df


if __name__ == "__main__":
    data = pd.read_csv(
        "1_data/aapl_preprocessed.csv",
        index_col="Date",
        parse_dates=True
    )

    data = feature_engineering(data)
print(data.shape)
print(data.shape) 