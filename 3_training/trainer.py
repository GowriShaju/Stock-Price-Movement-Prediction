import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import sys
import os

sys.path.append(os.path.abspath("2_models"))
from lstm import LSTM_model 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Hyperparameters
hidden_size   = 256  
num_layers    = 2
dropout       = 0.2
learning_rate = 0.001
batch_size    = 32
epochs        = 100
patience      = 15


# Dataset
class StockDataset(Dataset):
    def __init__(self, X, y_direction, y_magnitude):
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y_direction = torch.tensor(y_direction, dtype = torch.float32)
        self.y_magnitude = torch.tensor(y_magnitude, dtype = torch.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.y_direction[idx], self.y_magnitude[idx]

    def __len__(self):
        return len(self.X)


# Load Data
def load_data():
    X_train     = np.load("1_data/X_train.npy")
    y_dir_train = np.load("1_data/y_dir_train.npy")
    y_mag_train = np.load("1_data/y_mag_train.npy")

    X_val       = np.load("1_data/X_val.npy")
    y_dir_val   = np.load("1_data/y_dir_val.npy")
    y_mag_val   = np.load("1_data/y_mag_val.npy")

    print(f"Train: {X_train.shape} | Val: {X_val.shape}")

    up_days    = y_dir_train.sum()
    down_days  = len(y_dir_train) - up_days
    pos_weight = 1 

    print(f"Up: {int(up_days)} | Down: {int(down_days)} | pos_weight: {pos_weight:.4f}")
    print(f"Class balance: {y_dir_train.mean():.4f}")

    return (X_train, y_dir_train, y_mag_train,
            X_val,   y_dir_val,   y_mag_val,
            pos_weight)


# Train One Epoch
def train_one_epoch(model, loader, optimizer, bce_loss, mse_loss, device):
    model.train()
    total_loss = 0

    for X_batch, y_dir_batch, y_mag_batch in loader:
        X_batch     = X_batch.to(device)
        X_batch = X_batch + 0.01 * torch.randn_like(X_batch) 
        y_dir_batch = y_dir_batch.to(device)
        y_mag_batch = y_mag_batch.to(device)

        optimizer.zero_grad()

        direction, magnitude = model(X_batch)
        direction = direction / 0.5

        loss_dir = bce_loss(direction.squeeze(), y_dir_batch)
        loss_mag = mse_loss(magnitude.squeeze(), y_mag_batch)

        # balanced loss
        loss = loss_dir + 0.5 * loss_mag

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# Evaluate
def evaluate(model, loader, bce_loss, mse_loss, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_dir_batch, y_mag_batch in loader:
            X_batch     = X_batch.to(device)
            y_dir_batch = y_dir_batch.to(device)
            y_mag_batch = y_mag_batch.to(device)

            direction, magnitude = model(X_batch)
            direction = direction / 0.5

            loss_dir = bce_loss(direction.squeeze(), y_dir_batch)
            loss_mag = mse_loss(magnitude.squeeze(), y_mag_batch)

            # SAME loss
            loss = loss_dir + 0.5 * loss_mag
            total_loss += loss.item()

    return total_loss / len(loader)


# Accuracy
def calculate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for X_batch, y_dir_batch, _ in loader:
            X_batch     = X_batch.to(device)
            y_dir_batch = y_dir_batch.to(device)

            direction, _ = model(X_batch)

            probs = torch.sigmoid(direction.squeeze())
            predicted = (probs > 0.5).float()

            correct += (predicted == y_dir_batch).sum().item()
            total   += y_dir_batch.size(0)

    return correct / total


# Check Predictions
def check_predictions(model, X_val, device):
    model.eval()
    with torch.no_grad():
        X_sample = torch.tensor(X_val[:200], dtype = torch.float32).to(device)
        direction, magnitude = model(X_sample)
        probs = torch.sigmoid(direction.squeeze()).cpu().numpy()
        mags  = magnitude.squeeze().cpu().numpy()

        print("\nPrediction distribution:")
        print(f"Min : {probs.min():.4f}")
        print(f"Max : {probs.max():.4f}")
        print(f"Mean: {probs.mean():.4f}")
        print(f">0.5: {(probs > 0.5).sum()}")
        print(f"<0.5: {(probs < 0.5).sum()}")

        print("\nMagnitude prediction:")
        print(f"Min : {mags.min():.4f}")
        print(f"Max : {mags.max():.4f}")
        print(f"Mean: {mags.mean():.4f}\n")


# Training
def train(hidden_size, learning_rate, dropout):

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    (X_train, y_dir_train, y_mag_train,
     X_val,   y_dir_val,   y_mag_val,
     pos_weight) = load_data()

    train_dataset = StockDataset(X_train, y_dir_train, y_mag_train)
    val_dataset   = StockDataset(X_val,   y_dir_val,   y_mag_val)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, generator=torch.Generator().manual_seed(42))
    val_loader   = DataLoader(val_dataset,   batch_size = batch_size, shuffle = False)

    input_size = X_train.shape[2]

    model = LSTM_model(
        input_size  = input_size,
        hidden_size = hidden_size,
        num_layers  = num_layers,
        dropout     = dropout
    ).to(device)

    print(f"Input size: {input_size}")

    pos_weight_tensor = torch.tensor([pos_weight], dtype = torch.float32).to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight = pos_weight_tensor)
    mse_loss = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = learning_rate,
        weight_decay = 1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = 'min',
        factor = 0.5,
        patience = 5
    )

    best_val_loss  = float('inf')
    patience_count = 0
    best_epoch     = 0

    print("\nTraining started...\n")

    for epoch in range(epochs):

        train_loss = train_one_epoch(model, train_loader, optimizer, bce_loss, mse_loss, device)
        val_loss   = evaluate(model, val_loader, bce_loss, mse_loss, device)
        train_acc = calculate_accuracy(model, train_loader, device)
        val_acc   = calculate_accuracy(model, val_loader, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.3f} | "
              f"Val Acc: {val_acc:.3f} | "
              f"LR: {current_lr:.6f}")

        if epoch == 0 or epoch % 5 == 0:
            check_predictions(model, X_val, device)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0 
            best_epoch     = epoch + 1
            torch.save(model.state_dict(), "artifacts/models/best_model.pth")
            print("Best model saved")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f}")
                break

    print(f"\nTraining complete!")
    print(f"Best Val Loss : {best_val_loss:.4f}")
    print(f"Best Epoch    : {best_epoch}")

    return best_val_loss


if __name__ == "__main__":
   train(hidden_size, learning_rate, dropout)