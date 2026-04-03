import numpy as np
import pandas as pd


def create_windows(df, window_size = 30):
    target_cols  = ["target_direction", "target_magnitude"]
    feature_cols = [
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

    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    features = df[feature_cols].values
    y_dir = df["target_direction"].values
    y_mag = df["target_magnitude"].values

    X = []
    y_direction = []
    y_magnitude = []

    for i in range(len(df) - window_size):
        window = features[i : i + window_size]

        if np.isnan(window).any() or np.isinf(window).any():
            continue

        target_dir = y_dir[i + window_size]
        target_mag = y_mag[i + window_size]

        X.append(window)
        y_direction.append(target_dir)
        y_magnitude.append(target_mag)

    X  = np.array(X, dtype=np.float32)
    y_direction = np.array(y_direction, dtype = np.float32)
    y_magnitude = np.array(y_magnitude, dtype = np.float32)

    return X, y_direction, y_magnitude


def save_windows(X, y_direction, y_magnitude, name):
    np.save(f"1_data/X_{name}.npy",     X)
    np.save(f"1_data/y_dir_{name}.npy", y_direction)
    np.save(f"1_data/y_mag_{name}.npy", y_magnitude)

    print(f"\nSaved {name}:")
    print(f"  X           : {X.shape}")
    print(f"  y_direction : {y_direction.shape}")
    print(f"  y_magnitude : {y_magnitude.shape}")


train = pd.read_csv("1_data/train.csv", index_col = "Date", parse_dates = True)
val   = pd.read_csv("1_data/val.csv",   index_col = "Date", parse_dates = True)
test  = pd.read_csv("1_data/test.csv",  index_col = "Date", parse_dates = True)

print(f"Train: {train.shape} | Val: {val.shape} | Test: {test.shape}")

X_train, y_dir_train, y_mag_train = create_windows(train)
X_val,   y_dir_val,   y_mag_val   = create_windows(val)
X_test,  y_dir_test,  y_mag_test  = create_windows(test)

save_windows(X_train, y_dir_train, y_mag_train, "train")
save_windows(X_val,   y_dir_val,   y_mag_val,   "val")
save_windows(X_test,  y_dir_test,  y_mag_test,  "test")

print("\nTrain direction distribution:")
print(np.unique(y_dir_train, return_counts = True))

import numpy as np

print("Train distribution:", np.mean(y_dir_train))
print("Val distribution:", np.mean(y_dir_val))
print("Train UP %:", y_dir_train.mean())
print("Val UP %:", y_dir_val.mean())