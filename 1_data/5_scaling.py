import pandas as pd

def split_data(df, train_ratio = 0.7, val_ratio = 0.15):

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * val_ratio) + train_end

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    print("Splitting done \n")
    print("n:", n)
    print("Train length:", len(train))
    print("Test length:", len(test))
    print("Val length:", len(val))

    return train, val, test


def scale_data(df, train, val, test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    target_cols = ['target_magnitude', 'target_direction']
    features = [
    "return_1",
    "momentum_3",
    "momentum_7",
    "momentum_14",
    "momentum_3_smooth",
    "trend_20",
    "trend_30",
    "ema_diff",
    "volatility_10",
    "price_position",
    "vol_spike",
    "volatility_3",
    "return_acc",
    "momentum_change",
    "price_norm"
]
    print(f"Feature Columns: {features}")

    train[features] = scaler.fit_transform(train[features])
    val[features]   = scaler.transform(val[features])
    test[features]  = scaler.transform(test[features])

    return train, val, test, scaler, features


def save_scaler(scaler):
    import pickle
    with open("1_data/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved")


def save_splits(train, val, test):
    train.to_csv('1_data/train.csv', index = True)
    val.to_csv('1_data/val.csv', index = True)
    test.to_csv('1_data/test.csv', index = True)

    return  train, val, test



data = pd.read_csv('1_data/aapl_final.csv', index_col = 'Date', parse_dates = True)
train, val, test = split_data(data, train_ratio = 0.7, val_ratio = 0.15)
train, val, test, scaler, features = scale_data(data, train, val, test)
save_scaler(scaler)
save_splits(train, val, test)
save_splits(train, val, test)
