# app.py — HR Retention Dashboard (single file)
# ---------------------------------------------
# How to run locally:
#   pip install streamlit plotly pandas numpy scikit-learn
#   streamlit run app.py
#
# Works on Streamlit Community Cloud (point it at this file in your GitHub repo).
# Upload hr_data.csv when prompted, or click "Load sample data" to try it instantly.

import io
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="HR Retention Dashboard", layout="wide")

# ====================== Helpers ======================

def make_sample_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    departments = ["Sales","Accounting","HR","Technical","Support","IT","Product Management","Marketing","Management","RandD"]
    salary = rng.choice(["low","medium","high"], size=n, p=[0.45,0.4,0.15])
    dept = rng.choice(departments, size=n)
    satisfaction = np.clip(rng.normal(0.65, 0.18, size=n), 0, 1)
    evals = np.clip(rng.normal(0.72, 0.12, size=n), 0, 1)
    projects = rng.integers(2, 7, size=n)
    hours = np.clip(rng.normal(200, 25, size=n) + (projects-4)*10, 90, 310)
    tenure = np.clip(rng.normal(3.6, 1.8, size=n), 0.5, 10).round(1)
    accident = rng.choice([0,1], size=n, p=[0.95,0.05])
    promo5 = rng.choice([0,1], size=n, p=[0.96,0.04])

    # Base log-odds driven by realistic factors
    logit = (
        -3.0*(satisfaction-0.65)                  # satisfaction protective
        + 0.45*(hours-200)/25                     # workload risk
        + 0.35*(tenure-4.0)                       # stagnation with tenure
        + 0.20*(projects-4)                       # more projects more risk
        - 0.6*promo5                              # promotion protects
        - 0.2*accident                            # post-incident attention
    )
    # salary effects
    logit += np.where(salary=="low", 1.1, 0)
    logit += np.where(salary=="medium", 0.6, 0)
    # department bumps
    logit += np.where(np.isin(dept, ["HR","Accounting","Technical","IT"]), 0.3, 0)
    prob = 1/(1+np.exp(-logit))
    left = rng.binomial(1, np.clip(prob, 0.01, 0.95))
    df = pd.DataFrame({
        "satisfaction_level": satisfaction,
        "last_evaluation": evals,
        "number_project": projects,
        "average_montly_hours": hours.astype(int),
        "time_spend_company": tenure,
        "Work_accident": accident,
        "promotion_last_5years": promo5,
        "Department": dept,
        "salary": salary,
        "left": left
    })
    return df

def tenure_band(y):
    if y < 1: return "0–1"
    elif y < 3: return "1–3"
    elif y < 5: return "3–5"
    else: return "5+"

def hours_band(h):
    if h < 150: return "<150"
    elif h < 180: return "150–179"
    elif h < 210: return "180–209"
    elif h < 240: return "210–239"
    else: return "240+"

@st.cache_data(show_spinner=False)
def load_df(file_bytes: bytes | None) -> pd.DataFrame:
    if file_bytes is None:
        return None
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df

@st.cache_data(show_spinner=False)
def prep_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tenure_band"] = df["time_spend_company"].apply(tenure_band)
    df["hours_band"]  = df["average_montly_hours"].apply(hours_band)
    return df

@st.cache_resource(show_spinner=False)
def build_model(df: pd.DataFrame):
    num = ["satisfaction_level","last_evaluation","number_project",
           "average_montly_hours","time_spend_company","Work_accident","promotion_last_5years"]
    cat = ["Department","salary"]
    X, y = df[num + cat], df["left"]

    preproc = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat)
    ])
    clf = Pipeline([("prep", preproc),
                    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:,1])

    # Coefficients (standardized space)
    ohe = clf.named_steps["prep"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(cat)
    feat_names = np.concatenate([num, cat_names])
    coefs = clf.named_steps["model"].coef_.ravel()
    coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df.sort_values("abs_coef", ascending=False, inplace=True)
    return clf, auc, coef_df, num, cat

def apply_filters(df: pd.DataFrame, dept, salary, eval_min, hours_range, tenure_range):
    sub = df.copy()
    if dept != "All":
        sub = sub[sub["Department"] == dept]
    if salary != "All":
        sub = sub[sub["salary"] == salary]
    sub = sub[sub["last_evaluation"] >= eval_min]
    sub = sub[(sub["average_montly_hours"].between(*hours_range)) &
              (sub["time_spend_company"].between(*tenure_range))]
    return sub

def kpi_card(col, label, value):
    if value is None or np.isnan(value):
        col.metric(label, "–")
    else:
        col.metric(label, f"{value:.1%}")

# ====================== UI: Data load ======================

st.title("HR Retention Dashboard")
st.caption("Interactive analytics for turnover drivers, risk hotspots, and targeted interventions.")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload hr_data.csv", type=["csv"])
    use_sample = st.button("Load sample data")

if uploaded is not None:
    df_in = load_df(uploaded.read())
elif use_sample:
    df_in = make_sample_data()
else:
    st.info("Upload **hr_data.csv**, or click **Load sample data** to explore the dashboard.")
    st.stop()

df = prep_bands(df_in)

# ====================== Sidebar Filters ======================

with st.sidebar:
    st.header("Filters")
    departments = ["All"] + sorted(df["Department"].unique().tolist())
    dept_sel = st.selectbox("Department", departments, index=0)
    salary_sel = st.selectbox("Salary", ["All","low","medium","high"], index=0)
    eval_min = st.slider("Minimum Evaluation", 0.0, 1.0, 0.0, 0.05)
    hours_min, hours_max = int(df["average_montly_hours"].min()), int(df["average_montly_hours"].max())
    hours_range = st.slider("Monthly Hours Range", hours_min, hours_max, (max(90, hours_min), min(310, hours_max)), 5)
    tenure_min, tenure_max = int(max(1, df["time_spend_company"].min())), int(df["time_spend_company"].max())
    tenure_range = st.slider("Tenure Range (years)", tenure_min, max(tenure_max, tenure_min+1), (min(1, tenure_min), min(7, tenure_max)), 1)
    model_salary = st.selectbox("Model Salary (risk heatmap)", ["low","medium","high"], index=1)

# ====================== Model ======================

with st.spinner("Training driver model…"):
    clf, auc, coef_df, num_cols, cat_cols = build_model(df)

st.write(f"**Driver Model ROC AUC:** `{auc:.3f}`  — higher is better (0.5 = random, 1.0 = perfect).")

# ====================== KPIs ======================

sub = apply_filters(df, dept_sel, salary_sel, eval_min, hours_range, tenure_range)
c1, c2, c3, c4 = st.columns(4)
kpi_card(c1, "Turnover Rate", sub["left"].mean() if len(sub) else np.nan)
kpi_card(c2, "High-Performer Flight (eval≥0.8)", sub.loc[sub["last_evaluation"]>=0.8,"left"].mean() if len(sub) else np.nan)
kpi_card(c3, "Burnout Signal (≥210h/mo)", (sub["average_montly_hours"]>=210).mean() if len(sub) else np.nan)
kpi_card(c4, "Stagnation (≥4yrs & no promo)", ((sub["time_spend_company"]>=4) & (sub["promotion_last_5years"]==0)).mean() if len(sub) else np.nan)

st.divider()

# ====================== Charts (filtered) ======================

if len(sub) == 0:
    st.warning("No data matches the current filters.")
else:
    # Turnover heatmap: Hours × Tenure
    heat = sub.pivot_table(index="hours_band", columns="tenure_band", values="left", aggfunc="mean")
    heat = heat.reindex(index=["<150","150–179","180–209","210–239","240+"],
                        columns=["0–1","1–3","3–5","5+"])
    fig_heat = px.imshow(
        heat, text_auto=".2f", aspect="auto",
        color_continuous_scale="Viridis",
        labels=dict(color="Turnover Rate", x="Tenure Band", y="Hours Band"),
        title="Turnover Rate by Hours × Tenure (filtered)"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Department turnover bar
    dept_turn = sub.groupby("Department")["left"].mean().sort_values(ascending=False).reset_index()
    fig_bar = px.bar(
        dept_turn, x="Department", y="left", text="left",
        labels={"left":"Turnover Rate"},
        title="Department Turnover Rate (filtered)"
    )
    fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Evaluation band turnover (U-shape)
    bins = [0,0.5,0.6,0.7,0.8,0.9,1.0]
    labels = ['≤0.5','0.5–0.6','0.6–0.7','0.7–0.8','0.8–0.9','0.9–1.0']
    sub_eval = sub.copy()
    sub_eval["evaluation_band"] = pd.cut(sub_eval["last_evaluation"], bins=bins, labels=labels, include_lowest=True)
    eval_turn = sub_eval.groupby("evaluation_band")["left"].mean().reindex(labels).reset_index()
    fig_eval = px.bar(
        eval_turn, x="evaluation_band", y="left", text="left",
        labels={"left":"Turnover Rate","evaluation_band":"Evaluation Band"},
        title="Turnover Rate by Performance Evaluation Band (filtered)"
    )
    fig_eval.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig_eval, use_container_width=True)

st.divider()

# ====================== Model-based risk heatmap (Tenure × Hours) ======================

# Create a prediction grid with median values for other features
tenure_vals = [0.5, 2, 4, 6]
hours_vals  = [150, 180, 210, 240]
dept_mode   = df["Department"].mode()[0]
med         = df[num_cols].median()

grid = []
for t in tenure_vals:
    for h in hours_vals:
        row = {
            "satisfaction_level": med["satisfaction_level"],
            "last_evaluation": med["last_evaluation"],
            "number_project": int(round(med["number_project"])),
            "average_montly_hours": h,
            "time_spend_company": t,
            "Work_accident": int(round(med["Work_accident"])),
            "promotion_last_5years": int(round(med["promotion_last_5years"])),
            "Department": dept_mode,
            "salary": st.session_state.get("model_salary_for_grid", "medium")
        }
        grid.append(row)

# Let user choose the salary tier for the prediction grid
st.session_state["model_salary_for_grid"] = st.selectbox(
    "Predicted Risk Heatmap — choose Salary tier",
    ["low","medium","high"], index=1
)
for r in grid:
    r["salary"] = st.session_state["model_salary_for_grid"]

grid = pd.DataFrame(grid)
grid["pred"] = clf.predict_proba(grid[num_cols + cat_cols])[:,1]
risk = grid.pivot_table(index="average_montly_hours", columns="time_spend_company", values="pred")
fig_model = px.imshow(
    risk, text_auto=".2f", aspect="auto",
    color_continuous_scale="Viridis",
    labels=dict(color="Predicted Turnover", x="Tenure (yrs)", y="Monthly Hours"),
    title=f"Predicted Turnover Probability by Tenure × Hours — Salary: {st.session_state['model_salary_for_grid']}"
)
st.plotly_chart(fig_model, use_container_width=True)

st.divider()

# ====================== Driver model: top drivers ======================

st.subheader("Top Drivers of Turnover (driver model)")
topn = coef_df.head(15).copy()
topn["direction"] = np.where(topn["coef"] >= 0, "↑ increases risk", "↓ reduces risk")
fig_coef = px.bar(
    topn.sort_values("abs_coef"),
    x="abs_coef", y="feature", orientation="h",
    color="direction",
    labels={"abs_coef":"Strength (|coefficient|)", "feature":"Feature"},
    title="Top 15 drivers by absolute coefficient"
)
st.plotly_chart(fig_coef, use_container_width=True)
st.caption("Note: Signs show direction; bar length shows strength. Satisfaction typically has a large negative coefficient → higher satisfaction strongly reduces turnover risk.")

# ====================== At-risk list ======================

st.subheader("At-Risk Employees (model predictions)")
df_pred = df.copy()
num_cols = ["satisfaction_level","last_evaluation","number_project","average_montly_hours",
            "time_spend_company","Work_accident","promotion_last_5years"]
cat_cols = ["Department","salary"]
df_pred["pred_prob"] = clf.predict_proba(df_pred[num_cols + cat_cols])[:,1]

df_table = apply_filters(df_pred, dept_sel, salary_sel, eval_min, hours_range, tenure_range)
N = st.slider("Show top N at-risk employees", 10, 200, 25, 5)
cols_show = ["Department","salary","satisfaction_level","last_evaluation","number_project",
             "average_montly_hours","time_spend_company","promotion_last_5years","pred_prob","left"]
st.dataframe(
    df_table.sort_values("pred_prob", ascending=False).head(N)[cols_show]
            .style.format({
                "pred_prob":"{:.2%}",
                "satisfaction_level":"{:.2f}",
                "last_evaluation":"{:.2f}",
                "average_montly_hours":"{:.0f}",
                "time_spend_company":"{:.1f}"
            }),
    use_container_width=True
)

st.caption("Tip: Use sidebar filters (Department, Salary, Evaluation, Hours, Tenure) to focus KPIs, charts, and the at-risk list on specific segments.")
