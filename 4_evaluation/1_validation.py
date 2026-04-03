import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import os

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
    plot_regression_metrics,
    plot_residuals
) 

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
    X_val = np.load("1_data/X_val.npy")
    y_dir_val = np.load("1_data/y_dir_val.npy")
    y_mag_val = np.load("1_data/y_mag_val.npy")
    return X_val, y_dir_val, y_mag_val


# Validate
def validate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_val, y_dir_val, y_mag_val = load_data()

    dataset = StockDataset(X_val, y_dir_val, y_mag_val)
    loader = DataLoader(dataset, batch_size = 32, shuffle = False)

    input_size = X_val.shape[2]

    model = LSTM_model(
        input_size = input_size,
        hidden_size = 256,
        num_layers = 2,
        dropout = 0.2
    ).to(device)

    model.load_state_dict(torch.load("2_models/best_model.pth", map_location=device))
    model.eval()

    all_probs = []
    all_labels = []
    all_mag_preds = []
    all_mag_true = []

    # Collect predictions 
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

    
    # Basic stats
    print("\n===== BASIC STATS =====")
    print(f"Mean Prob : {np.mean(all_probs):.4f}")
    print(f"Min Prob  : {np.min(all_probs):.4f}")
    print(f"Max Prob  : {np.max(all_probs):.4f}")
    print(f"UP (>0.5) : {(all_probs > 0.5).sum()}")
    print(f"DOWN (<0.5): {(all_probs < 0.5).sum()}")
    print(f"Actual UP %: {np.mean(all_labels):.4f}")

    
    # Threshold tuning
    from sklearn.metrics import f1_score, accuracy_score

    best_f1 = 0
    best_thresh = 0.5

    for t in np.linspace(0.1, 0.9, 50):
       preds = (all_probs > t).astype(int)
    
       if preds.sum() == 0 or preds.sum() == len(preds):
          continue
        
       f1 = f1_score(all_labels, preds)

       if f1 > best_f1:
           best_f1 = f1
           best_thresh = t

    print("\n===== THRESHOLD TUNING =====")
    print(f"Best Threshold: {best_thresh:.3f}")
    print(f"Best Accuracy : {best_f1:.4f}")

   
    # Final predictions
    final_preds = (all_probs > best_thresh).astype(int)
    final_acc = f1_score(all_labels, final_preds)

    print("\n===== FINAL RESULTS =====")
    print(f"Final Accuracy: {final_acc:.4f}")
    print(f"Predicted UP: {(final_preds == 1).sum()} | DOWN: {(final_preds == 0).sum()}")

   
    # Metrics
    evaluate_metrics(all_probs, all_labels, threshold = best_thresh)

    
    # VISUALIZATIONS (CLASSIFICATION)
    print("\nGenerating classification plots...")
    #plot_probability_distribution(all_probs)
    #plot_predictions_vs_actual(all_probs, all_labels)
    #plot_confusion_matrix(all_labels, all_probs, threshold=best_thresh)
    #plot_roc_curve(all_labels, all_probs)
    #plot_precision_recall(all_labels, all_probs)

    
    # VISUALIZATIONS (REGRESSION)
    print("\nGenerating regression plots...")
    #plot_regression_predictions(all_mag_true, all_mag_preds)
    #plot_regression_metrics(all_mag_true, all_mag_preds)
    #plot_residuals(all_mag_true, all_mag_preds)

    
    # Debug info
    print("\n===== DEBUG INFO =====")
    print("Val UP %:", np.mean(all_labels))


# Run 
if __name__ == "__main__":
    validate()
