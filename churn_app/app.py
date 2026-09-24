"""Telco customer churn predictor (Streamlit)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
import streamlit as st

import churn_core as core

HERE = Path(__file__).parent
MODEL_PATH = HERE / "churn_model.joblib"
DATA_PATH = HERE / "Telco-Customer-Churn.csv"

BAND_COLOR = {"Low": "#2E7D5B", "Medium": "#C98A00", "High": "#C0392B"}
INK, SLATE, MUTED = "#1F2D3A", "#4E6577", "#C9D3DB"

st.set_page_config(page_title="Telco churn predictor", page_icon="📡", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1250px;}
    .badge {display: inline-block; padding: .25rem .8rem; border-radius: 999px;
            color: white; font-weight: 600; font-size: .95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------- model loading
@st.cache_resource(show_spinner="Loading model...")
def load_artifact(path: str, mtime: float) -> dict:
    return joblib.load(path)  # mtime is only here so a retrained file busts the cache


def ensure_model() -> dict:
    if MODEL_PATH.exists():
        return load_artifact(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)

    st.title("Telco churn predictor")
    st.warning("No trained model found (`churn_model.joblib`).")
    if DATA_PATH.exists():
        st.write("`Telco-Customer-Churn.csv` is here. Training takes a few minutes.")
        if st.button("Train model now", type="primary"):
            import train_model

            with st.spinner("Ranking features and tuning models..."):
                train_model.train(DATA_PATH, MODEL_PATH, n_iter=10, verbose=False)
            st.rerun()
    else:
        st.write(
            "Put `Telco-Customer-Churn.csv` next to `app.py`, run `python train_model.py`, "
            "then reload this page."
        )
    st.stop()


# ------------------------------------------------------------------------------ charts
def gauge(p: float, thr: float, band: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "valueformat": ".0f",
                "font": {"size": 46, "color": BAND_COLOR[band]}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": BAND_COLOR[band], "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, thr * 75], "color": "#E6F1EB"},
                {"range": [thr * 75, thr * 100], "color": "#F7EBCB"},
                {"range": [thr * 100, 100], "color": "#F4D9D5"},
            ],
            "threshold": {"line": {"color": INK, "width": 3}, "thickness": 0.9, "value": thr * 100},
        },
    ))
    fig.update_layout(height=250, margin=dict(l=25, r=25, t=15, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": INK})
    return fig


def driver_chart(exp: pd.DataFrame) -> go.Figure:
    top = exp.head(6).iloc[::-1]
    pts = top["effect"] * 100
    fig = go.Figure(go.Bar(
        x=pts,
        y=[f"{core.label(f)}: {core.format_value(f, v)}" for f, v in zip(top["feature"], top["value"])],
        orientation="h",
        marker_color=[BAND_COLOR["High"] if e > 0 else BAND_COLOR["Low"] for e in pts],
        text=[f"{e:+.1f}" for e in pts],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(height=280, margin=dict(l=10, r=40, t=10, b=30),
                      xaxis={"title": "Change in score (percentage points)",
                             "zeroline": True, "zerolinecolor": INK},
                      font={"color": INK},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def importance_chart(art: dict) -> go.Figure:
    rank = pd.Series(art["ranking"]).sort_values().tail(15)
    chosen = set(art["features"])
    fig = go.Figure(go.Bar(
        x=rank.values, y=[core.label(f) for f in rank.index], orientation="h",
        marker_color=[SLATE if f in chosen else MUTED for f in rank.index],
    ))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=30),
                      xaxis_title="Drop in ROC-AUC when the feature is shuffled",
                      font={"color": INK}, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def k_chart(art: dict) -> go.Figure:
    t, k = art["k_table"], len(art["features"])
    fig = go.Figure(go.Scatter(x=t["top_k"], y=t["cv_roc_auc"], mode="lines+markers",
                               line={"color": SLATE}, marker={"size": 8}))
    chosen = t[t["top_k"] == k]
    if len(chosen):
        fig.add_trace(go.Scatter(x=chosen["top_k"], y=chosen["cv_roc_auc"], mode="markers",
                                 marker={"size": 14, "color": BAND_COLOR["High"]}, name="chosen"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=30), showlegend=False,
                      xaxis_title="Number of top features", yaxis_title="CV ROC-AUC",
                      font={"color": INK}, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def roc_chart(art: dict) -> go.Figure:
    r = art["roc"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": MUTED}))
    fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", line={"color": SLATE, "width": 3}))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=30), showlegend=False,
                      xaxis_title="False positive rate", yaxis_title="True positive rate",
                      font={"color": INK}, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def confusion_chart(art: dict) -> go.Figure:
    cm = np.array(art["confusion"])
    fig = go.Figure(go.Heatmap(
        z=cm, x=["No churn", "Churn"], y=["No churn", "Churn"], text=cm,
        texttemplate="%{text}", colorscale=[[0, "#EEF2F5"], [1, SLATE]], showscale=False,
    ))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=30),
                      xaxis={"title": "Predicted"},
                      yaxis={"title": "Actual", "autorange": "reversed"},
                      font={"color": INK}, paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ------------------------------------------------------------------------------- forms
def customer_form(art: dict) -> dict:
    d, rng, cats = art["defaults"], art["ranges"], art["categories"]
    inputs = {}
    cols = st.columns(2)
    for i, f in enumerate(core.input_features(art["features"])):
        with cols[i % 2]:
            name, key = core.label(f), f"in_{f}"
            if f == "tenure":
                inputs[f] = st.slider(name, int(rng[f][0]), int(rng[f][1]), int(d[f]), key=key)
            elif f == "MonthlyCharges":
                lo, hi = float(np.floor(rng[f][0])), float(np.ceil(rng[f][1]))
                inputs[f] = st.slider(name, lo, hi, float(round(d[f] * 2) / 2), step=0.5, key=key)
            elif f == "TotalCharges":
                inputs[f] = st.number_input(name, float(rng[f][0]), float(rng[f][1]) * 1.5,
                                            float(d[f]), step=10.0, key=key)
            elif f == "SeniorCitizen":
                choice = st.selectbox(name, ["No", "Yes"], index=int(d[f]), key=key)
                inputs[f] = int(choice == "Yes")
            else:
                opts = cats[f]
                inputs[f] = st.selectbox(name, opts, index=opts.index(d[f]), key=key)
    return inputs


def show_prediction(art: dict, inputs: dict) -> None:
    thr = art["threshold"]
    X = core.build_frame(pd.DataFrame([inputs]), art["features"])
    p = float(core.score(art, X)[0])
    band = core.risk_band(p, thr)

    st.subheader("Churn risk")
    st.plotly_chart(gauge(p, thr, band))
    verdict = {
        "High": "above the flag threshold, so this customer is flagged as likely to churn",
        "Medium": "close to the flag threshold, worth keeping an eye on",
        "Low": "well below the flag threshold",
    }[band]
    st.markdown(
        f'<span class="badge" style="background:{BAND_COLOR[band]}">{band} risk</span>'
        f"&nbsp; Score is {verdict} ({thr:.0%}).",
        unsafe_allow_html=True,
    )
    st.caption(
        "The model is trained on balanced (resampled) data, so treat the score as a way to rank "
        "customers by risk rather than an exact probability."
    )

    st.markdown("#### What is driving this score")
    st.plotly_chart(driver_chart(core.explain(art, inputs)))
    st.caption("Each bar compares this customer's value with the typical spread of customers: "
               "red pushes the score up, green pushes it down.")

    st.markdown("#### Suggested retention actions")
    for tip in core.recommendations(inputs, band):
        st.markdown(f"- {tip}")


# ------------------------------------------------------------------------------- pages
def model_tab(art: dict) -> None:
    m, thr = art["metrics"], art["threshold"]
    c = st.columns(4)
    c[0].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c[1].metric("F1 (churners)", f"{m['f1']:.3f}")
    c[2].metric("Recall (churners)", f"{m['recall']:.1%}")
    c[3].metric("Precision (churners)", f"{m['precision']:.1%}")
    st.caption(f"Measured on {art['n_test']:,} customers the model never saw, "
               f"using a flag threshold of {thr:.0%}.")

    st.subheader("Model comparison")
    lb = art["leaderboard"].rename(columns={
        "model": "Model", "cv_roc_auc": "CV ROC-AUC", "oof_f1": "CV F1", "threshold": "Threshold",
        "accuracy": "Test accuracy", "precision": "Test precision", "recall": "Test recall",
        "f1": "Test F1", "roc_auc": "Test ROC-AUC",
    })
    st.dataframe(lb.round(3), hide_index=True)
    st.caption(f"{art['model_name']} won on cross-validated F1 for churners. "
               "Thresholds and the winner were chosen on training data only.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Which features matter")
        st.plotly_chart(importance_chart(art))
        st.caption("Dark bars are the features the model uses.")
    with right:
        st.subheader("How many features are enough")
        st.plotly_chart(k_chart(art))
        st.caption("The smallest set that keeps ROC-AUC within a hair of the best was kept.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("ROC curve")
        st.plotly_chart(roc_chart(art))
    with right:
        st.subheader("Confusion matrix")
        st.plotly_chart(confusion_chart(art))


def batch_tab(art: dict) -> None:
    need = core.input_features(art["features"])
    thr = art["threshold"]
    st.write("Score many customers at once. Your CSV needs these columns: "
             + ", ".join(f"`{f}`" for f in need) + ".")

    template = pd.DataFrame([{f: art["defaults"][f] for f in need}])
    st.download_button("Download CSV template", template.to_csv(index=False),
                       "churn_template.csv", "text/csv")

    up = st.file_uploader("Customers CSV", type="csv")
    if up is None:
        return

    raw = pd.read_csv(up)
    df = core.clean_features(raw)
    missing = [f for f in need if f not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        return

    X = core.build_frame(df, art["features"])
    ok = ~X.isna().any(axis=1)
    if not ok.any():
        st.error("Every row has missing values in the columns the model needs.")
        return
    if not ok.all():
        st.warning(f"{int((~ok).sum())} rows with missing values were skipped.")

    scores = pd.Series(np.nan, index=raw.index)
    scores[ok] = core.score(art, X[ok])
    out = raw.copy()
    out["churn_score"] = scores.round(3)
    out["risk_level"] = scores.apply(lambda p: core.risk_band(p, thr) if pd.notna(p) else "n/a")
    out["flagged"] = scores >= thr
    out = out.sort_values("churn_score", ascending=False)

    flagged = int(out["flagged"].sum())
    c = st.columns(3)
    c[0].metric("Customers scored", f"{int(ok.sum()):,}")
    c[1].metric("Flagged as likely to churn", f"{flagged:,}")
    c[2].metric("Share flagged", f"{flagged / ok.sum():.1%}")
    st.dataframe(out, hide_index=True)
    st.download_button("Download scored customers", out.to_csv(index=False),
                       "scored_customers.csv", "text/csv", type="primary")


# --------------------------------------------------------------------------------- main
art = ensure_model()

with st.sidebar:
    st.subheader("Model in use")
    st.write(f"**{art['model_name']}**, trained on {art['n_train']:,} customers")
    st.metric("ROC-AUC on test data", f"{art['metrics']['roc_auc']:.3f}")
    st.metric("Churners caught (recall)", f"{art['metrics']['recall']:.0%}")
    st.caption("Inputs the model uses, most influential first:")
    for f in art["features"]:
        st.markdown(f"- {core.label(f)}")
    if art.get("sklearn_version") != sklearn.__version__:
        st.warning(f"Model was trained with scikit-learn {art.get('sklearn_version')} but this "
                   f"app runs {sklearn.__version__}. Retrain if you see errors.")

st.title("Telco churn predictor")
st.write("Enter a customer's details to see how likely they are to leave, why, "
         "and what to do about it.")

tab_predict, tab_model, tab_batch = st.tabs(["Predict", "Model and features", "Batch scoring"])
with tab_predict:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("Customer profile")
        customer = customer_form(art)
    with right:
        show_prediction(art, customer)
with tab_model:
    model_tab(art)
with tab_batch:
    batch_tab(art)
