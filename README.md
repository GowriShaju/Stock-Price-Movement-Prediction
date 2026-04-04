# Stock Price Movement Prediction using LSTM

## Project Overview
This project focuses on predicting short-term stock price movements using historical financial data and deep learning techniques. The model leverages a Long Short-Term Memory (LSTM) network to capture temporal dependencies in stock price data.

The system is designed to perform **two tasks simultaneously**:
- **Classification** → Predict direction of movement (Up/Down)
- **Regression** → Predict magnitude of price change

This dual-task formulation provides a more comprehensive understanding of market behavior.

---

## Business Context
Stock price prediction is essential in:
- Algorithmic trading
- Portfolio management
- Risk assessment

Short-term forecasts help traders make informed decisions. However, financial data is:
- Noisy
- Non-linear
- Highly volatile

This project simulates real-world financial modeling challenges.

---

## Dataset Description
- **Source:** Yahoo Finance
- **Type:** Time-series data (daily)
- **Stock Used:** Reliance (or selected stock)

### Features:
- Open  
- High  
- Low  
- Close  
- Volume  

### Data Characteristics:
- Temporal dependency  
- Non-stationarity  
- Noise and volatility  

---

## Data Preprocessing
- Handled missing values (forward fill / removal)  
- Maintained chronological order (to avoid data leakage)  
- Applied **MinMax Scaling** for normalization  

---

## Feature Engineering
Along with raw features, the following indicators were created:
- Moving averages  
- Momentum features  
- Trend indicators  
- Volatility measures  

These help the model capture:
- Market trends  
- Momentum  
- Price dynamics  

---

## Target Engineering  

Instead of using simple day-to-day price changes, the targets are defined using **future returns over a fixed horizon**, making the problem a true forecasting task.

---

### Target Magnitude (Regression Task)  

Defined as the **future return over a horizon \( h \)**:

\[
\text{target\_magnitude} = \frac{P_{t+h}}{P_t} - 1
\]

**Represents:**
- The **percentage change in price** from current time \( t \) to future time \( t+h \)  
- The **strength and scale of price movement**  

---

### Target Direction (Classification Task)  

Derived from the **sign of the future return**:

\[
\text{target\_direction} =
\begin{cases}
1 & \text{if } \text{future return} > 0 \\
0 & \text{otherwise}
\end{cases}
\]

**Represents:**
- Whether the price will **increase (Up)** or **decrease (Down)** over the prediction horizon  
- A **binary interpretation** of the same movement captured in magnitude  

---

### Why this approach?

- Ensures **true future prediction** (no leakage from current values)  
- Maintains **consistency between regression and classification targets**  
- Avoids misleading formulations based on current-state indicators  
- Aligns with real-world financial modeling:
  - **Direction → trading decision (buy/sell)**
  - **Magnitude → expected return (risk/reward)**  

---

### Key Advantage  

Both targets are derived from the **same underlying quantity (future return)**, ensuring:

- Coherent learning  
- Better interpretability  
- Improved model stability  

---

### Summary  

| Target | Type | Meaning |
|--------|------|--------|
| `target_magnitude` | Regression | Future percentage return |
| `target_direction` | Classification | Sign of future return (Up/Down) |

---

This formulation transforms the problem from **trend detection** to a proper **time-series forecasting task**, making the model outputs more realistic and actionable.
---

## Problem Formulation

### 1. Classification Task
Predict:
- Whether the stock is moving **above or below trend**

### 2. Regression Task
Predict:
- Magnitude of price change (difference between consecutive prices)

---

## Sequence Generation
Used **Sliding Window Technique**

Example:
    (X[t-n], ..., X[t-1]) → X[t]

---

## Model Architecture

- LSTM Layers  
- Dropout (for regularization)  
- Fully Connected Layers  

### Outputs:
- Sigmoid → Classification  
- Linear → Regression  

---

## Training Details

- **Optimizer:** Adam  

- **Loss Functions:**
  - Binary Cross Entropy (Classification)  
  - Mean Squared Error (Regression)  

Combined Loss:
    L = λ1 * L_classification + λ2 * L_regression

---

## Regularization Techniques
- Dropout  
- Early Stopping  

---

## Data Splitting Strategy

Used **chronological split (NOT random)**:
- Training set  
- Validation set  
- Test set  

Prevents **future data leakage**.

---

## Hyperparameter Tuning

Tuned:
- Number of LSTM layers  
- Hidden units  
- Learning rate  
- Batch size  
- Dropout rate  

---

## Evaluation Metrics

### Classification:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- Confusion Matrix  

### Regression:
- MAE  
- MSE  
- RMSE  

---

## Results

### Classification:
- Accuracy: 0.8044  
- Precision: 0.8410  
- Recall: 0.8213  
- F1 Score: 0.8311  
- ROC-AUC: 0.8641  

### Regression:
- MAE: 0.0804  
- MSE: 0.0094  
- RMSE: 0.0969  

---

## Key Observations

- Strong performance in **trend prediction**  
- Slight bias toward upward movement  
- Regression struggles in high volatility  
- Model smooths extreme fluctuations  

---

## Limitations

- Sensitive to volatility  
- Limited feature set  
- Class imbalance bias  
- High computational cost  
- Difficulty capturing sudden spikes  

---

## Future Improvements

- Sentiment analysis integration  
- Transformer / Attention models  
- Ensemble methods  
- Real-time prediction system  
- Advanced hyperparameter tuning  

---

## Project Structure

    ├── data/
    |   ├── data_downloading.py
    |   ├── preprocessing.py
    |   ├── feature_engineering.py
    |   ├── target_columns.py
    |   ├── scaling.py
    |   ├── windowing.py
    ├── models/
    |   ├── lstm.py 
    ├── training/
    │   ├── trainer.py
    │   ├── tuning.py
    ├── evaluation/
    │   ├── metrics.py
    │   └── visualize.py
    |   ├── validation.py
    |   ├── test.py
    ├── results/
    |   ├── plots/
    |   ├── metrics_summar.txt
    ├── notebooks/
    |   ├── stock_price_movement_prediction.ipynb
    ├── artifacts/
    |   ├── models/
    |   ├── scalers/
    ├── README.md

---

## Setup Instructions

Clone repository:
    git clone <your-repo-link>
    cd <repo-name>

Install dependencies:
    pip install -r requirements.txt

Run training:
    python train.py

---

## Requirements

- Python  
- PyTorch  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  

---

## Reproducibility

- Fixed random seeds  
- Deterministic setup  

---

## Deliverables Included

- Clean dataset  
- LSTM implementation  
- Training pipeline  
- Evaluation metrics  
- Visualizations  
- Detailed report  

---

## Author

**Gowri Shaju**  
AI/ML Engineer(L1) Candidate  

---

## 🔹 Final Note

This project highlights the importance of **target engineering using rolling mean**, which improves stability and aligns predictions with real trading strategies instead of noisy day-to-day movements.
