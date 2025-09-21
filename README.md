# Credit Card Fraud Detection Dashboard

Detect fraudulent transactions using XGBoost and visualize results in an interactive Streamlit dashboard.

## Features:
- Preprocessing with StandardScaler
- Class imbalance handled using SMOTE
- Model: XGBoost
- Metrics: ROC-AUC 0.98, PR-AUC 0.86
- Interactive dashboard to predict fraud probability and highlight top 15 high-risk transactions

### Prerequisites
- Python 3.9+
- Libraries: pandas, xgboost, scikit-learn, shap (optional), joblib, streamlit, matplotlib


Install dependencies with:

```bash
pip install -r requirements.txt
