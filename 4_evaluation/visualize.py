import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error
)


SAVE_DIR = "5_results/plots"
os.makedirs(SAVE_DIR, exist_ok=True)


# 1. Probability Distribution
def plot_probability_distribution(probs):
    plt.figure()
    plt.hist(probs, bins = 50, color = 'steelblue', edgecolor = 'black')
    plt.title("Prediction Probability Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.savefig(f"{SAVE_DIR}/pred_prob_distribution.png")
    plt.close()



# 2. Predictions vs Actual (Classification)
def plot_predictions_vs_actual(probs, labels, n = 200):
    preds = (probs > 0.5).astype(int)

    plt.figure()
    plt.plot(labels[:n], label = "Actual", marker = 'o', color = 'skyblue')
    plt.plot(preds[:n], label = "Predicted", linestyle = '--', color = 'black')
    plt.title(f"Predicted vs Actual (First {n} samples)")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/act_vs_pred_200.png")
    plt.close()


# 3. Confusion Matrix
def plot_confusion_matrix(labels, probs, threshold = 0.5):
    preds = (probs > threshold).astype(int)
    cm = confusion_matrix(labels, preds)

    plt.figure()
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted") 
    plt.ylabel("Actual")
    plt.savefig(f"{SAVE_DIR}/confusion_matrix.png")
    plt.close()


# 4. ROC Curve
def plot_roc_curve(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label = f"AUC = {roc_auc:.3f}", color = 'navy')
    plt.plot([0,1], [0,1], linestyle = '--', color = 'hotpink')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/roc_curve.png")
    plt.close()



# 5. Precision-Recall Curve
def plot_precision_recall(labels, probs):
    precision, recall, _ = precision_recall_curve(labels, probs)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(f"{SAVE_DIR}/precision_recall_curve.png")
    plt.close()


# 6. Regression: Actual vs Predicted
def plot_regression_predictions(y_true, y_pred, n = 200):
    plt.figure()
    plt.plot(y_true[:n], label = "Actual", marker = 'o', mec = 'black')
    plt.plot(y_pred[:n], label = "Predicted", linestyle = '--', color = 'hotpink')
    plt.title("Regression: Actual vs Predicted")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/regression_act_vs_pred.png")
    plt.close()



# 7. Residual Plot
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals, color='skyblue', edgecolors='navy')
    plt.axhline(0, linestyle = '--', color = 'black')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.savefig(f"{SAVE_DIR}/residual_plot.png")
    plt.close()

