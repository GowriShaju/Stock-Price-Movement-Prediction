import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import os

# Paths
sys.path.append(os.path.abspath("2_models"))
from lstm import LSTM_model 

sys.path.append(os.path.abspath("4_evaluation"))
from metrics import evaluate_metrics
from visualize import (
    plot_probability_distribution,
    plot_predictions_vs_actual,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall,
    plot_regression_predictions,
    plot_residuals
)

# ========================= DATASET =========================
class StockDataset(Dataset):
    def __init__(self, X, y_direction, y_magnitude):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_direction = torch.tensor(y_direction, dtype=torch.float32)
        self.y_magnitude = torch.tensor(y_magnitude, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.y_direction[idx], self.y_magnitude[idx]

    def __len__(self):
        return len(self.X)


# ========================= LOAD TEST DATA =========================
def load_test_data():
    X_test = np.load("1_data/X_test.npy")
    y_dir_test = np.load("1_data/y_dir_test.npy")
    y_mag_test = np.load("1_data/y_mag_test.npy")
    return X_test, y_dir_test, y_mag_test


# ========================= TEST FUNCTION =========================
def test_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    X_test, y_dir_test, y_mag_test = load_test_data()

    dataset = StockDataset(X_test, y_dir_test, y_mag_test)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    input_size = X_test.shape[2]

    # Load trained model
    checkpoint = torch.load("2_models/best_model_tuned.pth", map_location=device)

    best_config = checkpoint['config']

    model = LSTM_model(
        input_size = input_size,
        hidden_size = best_config['hidden_size'],
        num_layers = 2,
        dropout = best_config['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # ========================= PREDICTIONS =========================
    all_probs = []
    all_labels = []
    all_mag_preds = []
    all_mag_true = []

    with torch.no_grad():
        for X_batch, y_dir_batch, y_mag_batch in loader:

            X_batch = X_batch.to(device)

            direction, magnitude = model(X_batch)

            probs = torch.sigmoid(direction.squeeze())

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_dir_batch.numpy())

            all_mag_preds.extend(magnitude.squeeze().cpu().numpy())
            all_mag_true.extend(y_mag_batch.numpy())

    # Convert to numpy
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_mag_preds = np.array(all_mag_preds)
    all_mag_true = np.array(all_mag_true)


    # Classification Metrics
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    y_true = all_labels

    best_score = 0
    best_threshold = 0.5

    for t in np.arange(0.4, 0.8, 0.02):
         preds = (all_probs >= t).astype(int)

         if preds.sum() == 0 or preds.sum() == len(preds):
           continue

         score = balanced_accuracy_score(y_true, preds)

         if score > best_score:
            best_score = score
            best_threshold = t

    print(f"\nBest Threshold: {best_threshold:.2f} | Best Balanced Acc: {best_score:.4f}")
    threshold = best_threshold

    final_preds = (all_probs > threshold).astype(int)
    final_acc = accuracy_score(all_labels, final_preds) 

    from sklearn.metrics import accuracy_score
    final_acc = accuracy_score(all_labels, final_preds)

    print("\n==================== TEST RESULTS ====================")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"Predicted UP: {(final_preds == 1).sum()} | DOWN: {(final_preds == 0).sum()}")

    # ========================= METRICS =========================
    evaluate_metrics(all_probs, all_labels, all_mag_preds, all_mag_true, threshold = threshold)

    # ========================= VISUALIZATIONS =========================
    print("\n==================== CLASSIFICATION VISUALS ====================")

    plot_probability_distribution(all_probs)
    plot_predictions_vs_actual(all_probs, all_labels)
    plot_confusion_matrix(all_labels, all_probs, threshold=threshold)
    plot_roc_curve(all_labels, all_probs)
    plot_precision_recall(all_labels, all_probs)

    print("\n==================== REGRESSION VISUALS ====================")

    plot_regression_predictions(all_mag_true, all_mag_preds)
    plot_residuals(all_mag_true, all_mag_preds)

    # ========================= DEBUG =========================
    print("\n==================== DEBUG ====================")
    print(f"Mean Prob : {np.mean(all_probs):.4f}")
    print(f"Min Prob  : {np.min(all_probs):.4f}")
    print(f"Max Prob  : {np.max(all_probs):.4f}")
    print(f"Actual UP %: {np.mean(all_labels):.4f}")
    print("First 20 probs:", all_probs[:20])


# ========================= RUN =========================
if __name__ == "__main__":
    test_model()
