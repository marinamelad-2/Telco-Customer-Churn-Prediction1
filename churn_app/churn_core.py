"""Streamlit-free logic shared by train_model.py and app.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Churn"
ID_COL = "customerID"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

LABELS = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges ($)",
    "TotalCharges": "Total charges ($)",
    "SeniorCitizen": "Senior citizen",
    "Contract": "Contract type",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "Partner": "Has partner",
    "Dependents": "Has dependents",
    "gender": "Gender",
}


def label(feature: str) -> str:
    return LABELS.get(feature, feature)


# --------------------------------------------------------------------------- data
def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning as the notebook, safe to run on raw training or new customer rows."""
    df = df.copy().drop(columns=[ID_COL], errors="ignore")

    if "SeniorCitizen" in df.columns and not pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = (
            df["SeniorCitizen"].astype(str).str.strip().str.lower()
            .map({"yes": 1, "no": 0, "1": 1, "0": 0})
        )

    if "TotalCharges" in df.columns:
        # The raw file stores 11 blanks (" ") for brand-new customers with tenure 0.
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        if {"tenure", "MonthlyCharges"} <= set(df.columns):
            df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])
    return df


def load_training_data(path) -> tuple[pd.DataFrame, pd.Series]:
    df = clean_features(pd.read_csv(path))
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    y = df.pop(TARGET).map({"Yes": 1, "No": 0}).astype(int)
    return df, y


# --------------------------------------------------------------------- model input
def input_features(features: list[str]) -> list[str]:
    """Features a user has to type in. TotalCharges is derived (tenure x monthly) when possible."""
    derivable = {"tenure", "MonthlyCharges"} <= set(features)
    return [f for f in features if not (f == "TotalCharges" and derivable)]


def build_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return the model's input columns, deriving TotalCharges when it was not supplied."""
    df = df.copy()
    if "TotalCharges" in features and "TotalCharges" not in df.columns:
        df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]
    return df[features]


def score(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    return artifact["pipeline"].predict_proba(X)[:, 1]


def risk_band(p: float, threshold: float) -> str:
    """High at/above the tuned threshold, Medium from 75% of it, otherwise Low."""
    if p >= threshold:
        return "High"
    if p >= 0.75 * threshold:
        return "Medium"
    return "Low"


# ------------------------------------------------------------------- explanations
def explain(artifact: dict, inputs: dict) -> pd.DataFrame:
    """Effect of each input on this customer's score versus a typical customer.

    For every input we swap it for values seen in a background sample of real customers and
    average the scores. effect = score_now - average_score_without_that_input, so a positive
    number means the input pushes this customer towards churning.
    """
    feats, bg = artifact["features"], artifact["background"]
    base = float(score(artifact, build_frame(pd.DataFrame([inputs]), feats))[0])

    rows = []
    for f in input_features(feats):
        rep = pd.concat([pd.DataFrame([inputs])] * len(bg), ignore_index=True)
        rep[f] = bg[f].to_numpy()
        avg = float(score(artifact, build_frame(rep, feats)).mean())
        rows.append({"feature": f, "value": inputs[f], "effect": base - avg})

    out = pd.DataFrame(rows)
    return out.sort_values("effect", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def format_value(feature: str, value) -> str:
    if feature == "SeniorCitizen":
        return "Yes" if int(value) == 1 else "No"
    if feature == "tenure":
        return f"{int(value)} mo"
    if feature in ("MonthlyCharges", "TotalCharges"):
        return f"${float(value):,.0f}"
    return str(value)


def recommendations(inputs: dict, band: str) -> list[str]:
    """Rule-based retention ideas. Only uses inputs the model actually collects."""
    if band == "Low":
        return ["Low risk: keep the relationship healthy, no special retention action needed."]

    tips = []
    if inputs.get("Contract") == "Month-to-month":
        tips.append("Offer a discount to move to a 1- or 2-year contract. Month-to-month customers churn the most.")
    if inputs.get("tenure") is not None and inputs["tenure"] < 12:
        tips.append("New customer: schedule an onboarding call and check-in during the first months.")
    if inputs.get("InternetService") == "Fiber optic":
        tips.append("Fiber customers churn more often. Check service quality and whether the price fits their usage.")
    if inputs.get("TechSupport") == "No":
        tips.append("Offer a free trial of tech support.")
    if inputs.get("OnlineSecurity") == "No":
        tips.append("Bundle online security at a reduced price.")
    if inputs.get("PaymentMethod") == "Electronic check":
        tips.append("Encourage automatic payment (bank transfer or card) with a small bill credit.")
    if inputs.get("MonthlyCharges") is not None and inputs["MonthlyCharges"] > 80:
        tips.append("High monthly bill: review whether a cheaper plan would suit them.")
    return tips[:4] or ["Reach out with a personal retention offer."]
