# Telco churn predictor (Streamlit)

## Run it
```bash
pip install -r requirements.txt
# 1) put Telco-Customer-Churn.csv in this folder, then train (a few minutes):
python train_model.py
# 2) start the app:
streamlit run app.py
```
You can also skip step 1: if `churn_model.joblib` is missing, the app shows a **Train model now** button.
Faster trial run: `python train_model.py --n-iter 5`.

Train and run with the same scikit-learn version (the app warns you if they differ).

## Files
| File | Role |
|---|---|
| `train_model.py` | Feature selection, model tuning, threshold tuning, saves `churn_model.joblib` |
| `churn_core.py` | Cleaning, scoring, per-customer explanation, retention tips (no Streamlit) |
| `app.py` | The Streamlit interface: Predict, Model and features, Batch scoring |

## How the "best features" and "best model" are chosen
1. **Features**: permutation importance ranks the 19 raw columns, then the smallest top-K set whose
   cross-validated ROC-AUC is within 0.003 of the best is kept. The form only asks for those inputs.
   `TotalCharges` is derived as tenure x monthly charges, so users do not type it.
2. **Model**: Logistic Regression, Random Forest, XGBoost and LightGBM are tuned with a random search.
   SMOTE runs inside the CV folds (in the notebook it ran before CV, which inflated CV scores).
3. **Threshold**: chosen per model from out-of-fold predictions to maximise F1 of the churn class.
   The winner is the model with the best out-of-fold F1. The test set is used once, for the final report.

Because the model trains on SMOTE-balanced data, the score ranks customers by risk; it is not a calibrated probability.
