import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import os

sys.path.append(os.path.abspath("2_models"))
from lstm import LSTM_model

from sklearn.metrics import accuracy_score


# ── Dataset ─────────────────────────
class StockDataset(Dataset):
    def __init__(self, X, y_direction, y_magnitude):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_direction = torch.tensor(y_direction, dtype = torch.float32)
        self.y_magnitude = torch.tensor(y_magnitude, dtype = torch.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.y_direction[idx], self.y_magnitude[idx]

    def __len__(self):
        return len(self.X)


# ── Load Data ───────────────────────
def load_data():
    X_train = np.load("1_data/X_train.npy")
    y_dir_train = np.load("1_data/y_dir_train.npy")
    y_mag_train = np.load("1_data/y_mag_train.npy")

    X_val = np.load("1_data/X_val.npy")
    y_dir_val = np.load("1_data/y_dir_val.npy")
    y_mag_val = np.load("1_data/y_mag_val.npy")

    return X_train, y_dir_train, y_mag_train, X_val, y_dir_val, y_mag_val


# ── Train One Config ─────────────────
def train_one(config, device):

    X_train, y_dir_train, y_mag_train, X_val, y_dir_val, y_mag_val = load_data()

    train_loader = DataLoader(
        StockDataset(X_train, y_dir_train, y_mag_train),
        batch_size = config["batch_size"],
        shuffle = True
    )

    val_loader = DataLoader(
        StockDataset(X_val, y_dir_val, y_mag_val),
        batch_size = 32,
        shuffle = False
    )

    model = LSTM_model(
        input_size =  X_train.shape[2],
        hidden_size = config["hidden_size"],
        num_layers = 2,
        dropout = config["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    # ── Training ──
    for epoch in range(config["epochs"]):
        model.train()

        for X_batch, y_dir_batch, y_mag_batch in train_loader:

            X_batch = X_batch.to(device)
            y_dir_batch = y_dir_batch.to(device)
            y_mag_batch = y_mag_batch.to(device)

            optimizer.zero_grad()

            direction, magnitude = model(X_batch)

            loss_dir = bce_loss(direction.view(-1), y_dir_batch.view(-1))
            loss_mag = mse_loss(magnitude.view(-1), y_mag_batch.view(-1))

            loss = loss_dir + 0.05 * loss_mag
            loss.backward()
            optimizer.step()

    # ── Validation ──
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_dir_batch, _ in val_loader:

            X_batch = X_batch.to(device)

            direction, _ = model(X_batch)
            probs = torch.sigmoid(direction.view(-1))

            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_dir_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)

    return acc, model


# ── Hyperparameter Search ────────────
def tune():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [
     {"hidden_size": 64,  "lr": 0.001,  "dropout": 0.2, "batch_size": 32, "epochs": 5},
     {"hidden_size": 128, "lr": 0.001,  "dropout": 0.2, "batch_size": 32, "epochs": 5},
     {"hidden_size": 128, "lr": 0.0003, "dropout": 0.3, "batch_size": 32, "epochs": 8},
     {"hidden_size": 256, "lr": 0.0005, "dropout": 0.3, "batch_size": 64, "epochs": 8},
]

    best_acc = 0
    best_config = None
    best_model = None

    print("\n--- Hyperparameter Tuning ---\n")

    for i, config in enumerate(configs):
        print(f"Running config {i+1}/{len(configs)}: {config}")

        acc, model = train_one(config, device)

        print(f"Validation Accuracy: {acc:.4f}\n")

        if acc > best_acc:
            best_acc = acc
            best_config = config
            best_model = model

    print("\n--- Best Result ---")
    print("Best Accuracy:", round(best_acc, 4))
    print("Best Config:", best_config)

    # Save best model
    torch.save({
    'model_state_dict': best_model.state_dict(),
    'config': best_config
}, "artifacts/models/best_model_tuned.pth")
    print("\nBest model saved as: best_model_tuned.pth")


if __name__ == "__main__":
    tune()