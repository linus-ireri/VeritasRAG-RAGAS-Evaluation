import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")
st.title("RAG Evaluation Dashboard")

RESULT_FILES = ["ragas_results.json", "ragas_results2.json"]

@st.cache_data
def load_json(path: str):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_records(data):
    rows = []
    for rec in data:
        if isinstance(rec, dict) and len(rec) == 1:
            inner = next(iter(rec.values()))
            if isinstance(inner, dict):
                rows.append(inner)
                continue
        rows.append(rec)
    return rows


def df_from_file(path):
    data = load_json(path)
    if not data:
        return None
    rows = flatten_records(data)
    return pd.DataFrame(rows)


# Sidebar: choose files to include
available = [f for f in RESULT_FILES if Path(f).exists() and Path(f).stat().st_size > 0]
if not available:
    st.warning("No result files found (look for ragas_results.json or ragas_results2.json)")
    st.stop()

selected = st.sidebar.multiselect("Select result files to include", options=available, default=available)

# Load and concatenate
frames = []
for f in selected:
    df = df_from_file(f)
    if df is not None:
        df["_source_file"] = f
        frames.append(df)

if not frames:
    st.error("No valid data loaded from selected files.")
    st.stop()

full = pd.concat(frames, ignore_index=True, sort=False)

# Compute numeric aggregates
numeric_cols = full.select_dtypes(include=["number"]).columns.tolist()
aggregates = {}
for col in numeric_cols:
    vals = full[col].dropna()
    # exclude NaN
    vals = vals[~vals.isna()]
    if len(vals) > 0:
        aggregates[col] = {
            "count": int(len(vals)),
            "mean": float(vals.mean()),
        }

# Top-level metrics
st.header("Aggregate Metrics")
cols = st.columns(max(1, len(aggregates)))
for i, (k, v) in enumerate(aggregates.items()):
    with cols[i % len(cols)]:
        st.metric(label=k, value=f"{v['mean']:.4f}", delta=f"{v['count']} valid")

# Plot averages bar chart
if aggregates:
    agg_df = pd.DataFrame({k: v["mean"] for k, v in aggregates.items()}, index=[0]).T
    agg_df.columns = ["mean"]
    st.subheader("Mean scores")
    st.bar_chart(agg_df)

# Per-sample table and filtering
st.header("Per-sample results")
filter_cols = [c for c in ["_source_file"] + numeric_cols if c in full.columns]
with st.expander("Filters and view options", expanded=True):
    src = st.selectbox("Source file", options=["All"] + sorted(full["_source_file"].unique().tolist()))
    if src != "All":
        view_df = full[full["_source_file"] == src]
    else:
        view_df = full
    show_n = st.number_input("Rows to show", min_value=10, max_value=len(view_df), value=25)

st.dataframe(view_df.head(show_n))

st.markdown("---")
st.markdown("Report generated from: " + ", ".join(selected))
