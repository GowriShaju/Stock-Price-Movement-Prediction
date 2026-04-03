import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_metrics(all_probs, all_labels,
                     all_mag_preds, all_mag_true,
                     threshold=0.5):

    # Ensure save folder exists
    os.makedirs("5_results", exist_ok=True)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ================= CLASSIFICATION =================
    preds = (all_probs > threshold).astype(int)

    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = 0.0

    try:
        ll = log_loss(all_labels, all_probs)
    except:
        ll = 0.0

    up_preds = (preds == 1).sum()
    down_preds = (preds == 0).sum()
    up_true = np.mean(all_labels)

    # ================= PRINT =================
    print("\n========== MODEL EVALUATION ==========")

    print("\n--- Classification Metrics ---")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\n--- Confusion Matrix ---")
    print(f"TP: {tp} | FP: {fp}")
    print(f"FN: {fn} | TN: {tn}")

    print("\n--- Probability Metrics ---")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Log Loss  : {ll:.4f}")

    print("\n--- Distribution ---")
    print(f"Predicted UP   : {up_preds}")
    print(f"Predicted DOWN : {down_preds}")
    print(f"Actual UP %    : {up_true:.4f}")


    # ================= REGRESSION =================

    mae = mean_absolute_error(all_mag_true, all_mag_preds)
    mse = mean_squared_error(all_mag_true, all_mag_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_mag_true, all_mag_preds)

    print("\n--- Regression Metrics ---")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    # ================= SAVE (WRITE) =================
    with open("5_results/metrics_summary.txt", "w") as f:
        f.write("========== MODEL EVALUATION ==========\n")

        f.write("\n--- Classification Metrics ---\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1 Score  : {f1:.4f}\n")

        f.write("\n--- Confusion Matrix ---\n")
        f.write(f"TP: {tp} | FP: {fp}\n")
        f.write(f"FN: {fn} | TN: {tn}\n")

        f.write("\n--- Probability Metrics ---\n")
        f.write(f"ROC-AUC   : {roc_auc:.4f}\n")
        f.write(f"Log Loss  : {ll:.4f}\n")

        f.write("\n--- Distribution ---\n")
        f.write(f"Predicted UP   : {up_preds}\n")
        f.write(f"Predicted DOWN : {down_preds}\n")
        f.write(f"Actual UP %    : {up_true:.4f}\n")

        f.write("\n--- Regression Metrics ---\n")
        f.write(f"MAE  : {mae:.4f}\n")
        f.write(f"MSE  : {mse:.4f}\n")
        f.write(f"RMSE : {rmse:.4f}\n")
        f.write(f"R2   : {r2:.4f}\n")

    print("\n======================================")

    # ================= RETURN =================
    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": ll,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }
    return results