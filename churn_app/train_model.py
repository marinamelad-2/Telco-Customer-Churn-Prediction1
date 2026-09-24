#!/usr/bin/env python
"""Train the Telco churn model and save everything the Streamlit app needs.

What it does
1. Loads and cleans the data (same cleaning as the notebook).
2. Ranks the raw features by permutation importance (on a validation slice of the train set).
3. Keeps the smallest top-K feature set whose cross-validated ROC-AUC is within --tol of the best.
4. Tunes Logistic Regression / Random Forest / XGBoost / LightGBM. SMOTE lives inside the
   pipeline, so it only touches the training folds (no leakage into validation folds).
5. Picks each model's decision threshold from out-of-fold predictions and chooses the winner by
   out-of-fold F1 of the churn class. The test set is only used for the final report.

Usage:  python train_model.py --data Telco-Customer-Churn.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import churn_core as core

try:
    from xgboost import XGBClassifier
except ImportError:  # optional: the model is simply skipped
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

SEED = 42


def make_pipeline(features: list[str], model) -> Pipeline:
    num = [c for c in features if c in core.NUMERIC]
    cat = [c for c in features if c not in core.NUMERIC]
    parts = []
    if num:
        parts.append(("num", StandardScaler(), num))
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return Pipeline([
        ("prep", ColumnTransformer(parts)),
        ("smote", SMOTE(random_state=SEED)),
        ("model", model),
    ])


def quick_rf() -> RandomForestClassifier:
    # Hyper-parameters close to the best ones found in the notebook.
    return RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_leaf=4, random_state=SEED, n_jobs=-1
    )


# ------------------------------------------------------------------ feature selection
def rank_features(X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED
    )
    pipe = make_pipeline(list(X_train.columns), quick_rf()).fit(X_fit, y_fit)
    imp = permutation_importance(
        pipe, X_val, y_val, scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=1
    )
    return pd.Series(imp.importances_mean, index=X_train.columns).sort_values(ascending=False)


def choose_k(X_train, y_train, ranking, cv, tol) -> tuple[int, pd.DataFrame]:
    n = len(ranking)
    ks = sorted({k for k in (3, 4, 5, 6, 8, 10, 12, 14) if k < n} | {n})
    rows = []
    for k in ks:
        feats = list(ranking.index[:k])
        auc = cross_val_score(
            make_pipeline(feats, quick_rf()), X_train[feats], y_train, cv=cv, scoring="roc_auc"
        )
        rows.append({"top_k": k, "cv_roc_auc": float(auc.mean())})
    table = pd.DataFrame(rows)
    best = table["cv_roc_auc"].max()
    k = int(table.loc[table["cv_roc_auc"] >= best - tol, "top_k"].min())
    return k, table


# ------------------------------------------------------------------------- modelling
def candidate_models() -> dict:
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000),
            {"model__C": loguniform(1e-2, 1e2)},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {
                "model__n_estimators": randint(150, 400),
                "model__max_depth": [4, 6, 8, 10, 12],
                "model__min_samples_leaf": randint(2, 15),
                "model__min_samples_split": randint(2, 20),
                "model__max_features": ["sqrt", 0.4, 0.6],
            },
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1, verbosity=0),
            {
                "model__n_estimators": randint(100, 400),
                "model__max_depth": randint(2, 7),
                "model__learning_rate": uniform(0.01, 0.19),
                "model__min_child_weight": randint(1, 10),
                "model__gamma": uniform(0, 5),
                "model__subsample": uniform(0.7, 0.3),
                "model__colsample_bytree": uniform(0.6, 0.4),
            },
        )
    if LGBMClassifier is not None:
        models["LightGBM"] = (
            LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1),
            {
                "model__n_estimators": randint(100, 400),
                "model__max_depth": [3, 4, 5, 6, 7],
                "model__learning_rate": uniform(0.01, 0.19),
                "model__min_child_samples": randint(15, 60),
                "model__num_leaves": randint(15, 60),
            },
        )
    return models


def best_threshold(y, proba) -> tuple[float, float]:
    grid = np.arange(0.20, 0.81, 0.01)
    scores = [f1_score(y, (proba >= t).astype(int)) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def report(pipe, X_test, y_test, thr) -> tuple[dict, np.ndarray, np.ndarray]:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= thr).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    return {k: float(v) for k, v in metrics.items()}, proba, pred


# ------------------------------------------------------------------------------ main
def train(data_path, out_path, n_iter: int = 15, tol: float = 0.003,
          top_k: int | None = None, verbose: bool = True) -> dict:
    warnings.filterwarnings("ignore")
    log = print if verbose else (lambda *a, **k: None)

    X, y = core.load_training_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    log(f"Data: {len(X):,} customers, churn rate {y.mean():.1%}, {X.shape[1]} raw features")

    # 1) features
    log("\n[1/3] Ranking features (permutation importance)...")
    ranking = rank_features(X_train, y_train)
    k, k_table = choose_k(X_train, y_train, ranking, cv, tol)
    if top_k:
        k = min(top_k, len(ranking))
    features = list(ranking.index[:k])
    log(k_table.round(4).to_string(index=False))
    log(f"-> keeping the top {k} features: {features}")

    # 2) models
    log("\n[2/3] Tuning models (SMOTE inside the CV folds)...")
    Xtr, Xte = X_train[features], X_test[features]
    fitted, rows = {}, []
    for name, (clf, params) in candidate_models().items():
        search = RandomizedSearchCV(
            make_pipeline(features, clf), params, n_iter=n_iter, scoring="roc_auc",
            cv=cv, random_state=SEED, n_jobs=1,
        ).fit(Xtr, y_train)
        oof = cross_val_predict(search.best_estimator_, Xtr, y_train, cv=cv,
                                method="predict_proba")[:, 1]
        thr, oof_f1 = best_threshold(y_train, oof)
        m, _, _ = report(search.best_estimator_, Xte, y_test, thr)
        fitted[name] = (search.best_estimator_, thr, search.best_params_)
        rows.append({"model": name, "cv_roc_auc": float(search.best_score_),
                     "oof_f1": oof_f1, "threshold": thr, **m})
        log(f"  {name:<20} CV AUC {search.best_score_:.4f} | OOF F1 {oof_f1:.4f} @ {thr:.2f}")

    leaderboard = pd.DataFrame(rows).sort_values("oof_f1", ascending=False).reset_index(drop=True)
    winner = leaderboard.loc[0, "model"]
    pipe, thr, params = fitted[winner]

    # 3) final report on the untouched test set
    log("\n[3/3] Final evaluation on the held-out test set")
    metrics, proba, pred = report(pipe, Xte, y_test, thr)
    log(f"Winner: {winner} (threshold {thr:.2f})")
    log({k: round(v, 4) for k, v in metrics.items()})

    fpr, tpr, _ = roc_curve(y_test, proba)
    idx = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))

    inputs = core.input_features(features)
    cats = [f for f in inputs if f not in core.NUMERIC]
    nums = [f for f in inputs if f in core.NUMERIC]
    artifact = {
        "pipeline": pipe,
        "model_name": winner,
        "best_params": params,
        "features": features,
        "threshold": thr,
        "metrics": metrics,
        "leaderboard": leaderboard,
        "ranking": ranking.to_dict(),
        "k_table": k_table,
        "roc": {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()},
        "confusion": confusion_matrix(y_test, pred).tolist(),
        "categories": {f: sorted(X_train[f].dropna().unique().tolist()) for f in cats},
        "ranges": {f: [float(X_train[f].min()), float(X_train[f].max())] for f in nums},
        "defaults": {
            **{f: float(X_train[f].median()) for f in nums},
            **{f: X_train[f].mode().iloc[0] for f in cats},
        },
        "background": X_train[inputs].sample(min(200, len(X_train)), random_state=SEED)
                                     .reset_index(drop=True),
        "churn_rate": float(y.mean()),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, out_path)
    log(f"\nSaved -> {out_path}")
    return artifact


if __name__ == "__main__":
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=str(here / "Telco-Customer-Churn.csv"))
    ap.add_argument("--out", default=str(here / "churn_model.joblib"))
    ap.add_argument("--n-iter", type=int, default=15, help="random-search trials per model")
    ap.add_argument("--tol", type=float, default=0.003,
                    help="ROC-AUC tolerance when choosing the smallest feature set")
    ap.add_argument("--top-k", type=int, default=None, help="force the number of features")
    a = ap.parse_args()
    train(a.data, a.out, n_iter=a.n_iter, tol=a.tol, top_k=a.top_k)
