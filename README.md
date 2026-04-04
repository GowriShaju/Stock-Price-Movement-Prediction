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
- **Stock Used:** Selected stock (e.g., Reliance / AAPL)

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

Targets are defined using **future returns over a fixed horizon**, making this a true forecasting problem.

### Target Magnitude (Regression Task)  

target_magnitude = Close(t+h) / Close(t) - 1

Represents:
- Future percentage price change  
- Strength of movement  

---

### Target Direction (Classification Task)  

target_direction = 1 if future_return > 0 else 0

Represents:
- Upward or downward movement over the prediction horizon  

---

### Why this approach?
- Ensures **true future prediction (no leakage)**  
- Keeps classification and regression **consistent**  
- Reflects real-world trading objectives  

---

## Problem Formulation

### Classification Task
Predict whether the price will go **Up or Down in the future**

### Regression Task
Predict the **magnitude of future return**

---

## Sequence Generation
Used **Sliding Window Technique**

(X[t-n], ..., X[t-1]) → X[t]

---

## Model Architecture

- LSTM Layers  
- Dropout (regularization)  
- Fully Connected Layers  

### Outputs:
- Sigmoid → Classification  
- Linear → Regression  

---

## Training Details

- **Optimizer:** Adam  

### Loss Functions:
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
- Training  
- Validation  
- Test  

Ensures **no future data leakage**.

---

## Hyperparameter Tuning

Tuned:
- Hidden units  
- Learning rate  
- Batch size  
- Dropout  

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
- **Accuracy:** ~0.60  
- **Precision:** ~0.62  
- **Recall:** ~0.74  
- **F1 Score:** ~0.68  
- **ROC-AUC:** ~0.59  

### Regression:
- **MAE:** ~0.06  
- **RMSE:** ~0.07  

---

## Key Observations

- The model achieves **~60% accuracy**, outperforming the naive baseline, indicating **meaningful predictive signal**.
- The model successfully captures **general directional trends** in stock movement.
- Balanced threshold tuning helped reduce **bias toward upward predictions**.
- Regression performance remains limited due to **high market volatility**.
- The model tends to **smooth extreme movements**, making sharp spikes harder to predict.
- Overall, the model reflects a **realistic and leakage-free financial forecasting setup**.

---

## Limitations

- Financial data is highly **noisy and unpredictable**  
- Limited ability to capture **sudden market shocks**  
- Weak regression performance (common in finance)  
- External factors (news, sentiment) not included  

---

## Future Improvements

- Incorporate sentiment analysis  
- Use Transformer-based models  
- Ensemble learning methods  
- Improve feature engineering  
- Add real-time prediction system  

---

## Project Structure

├── data/  
├── models/  
├── training/  
├── evaluation/  
├── results/  
├── notebooks/   
├── README.md  

---

## Setup Instructions

git clone <repo-link>  
cd <repo-name>  
pip install -r requirements.txt  
python train.py  

---

## Requirements

- Python  
- PyTorch  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib
- seaborn

---

## Final Note  

This project highlights the inherent difficulty of **financial time-series prediction**.  

Due to market noise, non-stationarity, and external influences, achieving very high accuracy is unrealistic.  

However, the model demonstrates that even **moderate performance (~60% accuracy, ~0.59 ROC-AUC)** can indicate **useful predictive capability** when the problem is formulated correctly and evaluated without bias.

---

## Author

**Gowri Shaju**  
AI/ML Engineer (L1) Candidate  
