import os, io, json, requests
import pandas as pd
import streamlit as st
import joblib

APP_TITLE = "Real Time Socring App (powered by SAS)"  # title as requested

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# ---------------------------
# Load model / metadata
# ---------------------------
def load_model():
    """
    Loads a scikit-learn pipeline from ./models/pipeline.pkl or MODEL_URL.
    Also tries to recover original feature lists from the preprocessor.
    Returns: (pipeline, numeric_cols, categorical_cols)
    """
    local_path = "models/pipeline.pkl"
    model_url = os.getenv("MODEL_URL")  # e.g. RAW GitHub URL

    pipe = None
    if os.path.isfile(local_path):
        pipe = joblib.load(local_path)
    elif model_url:
        r = requests.get(model_url, timeout=30)
        r.raise_for_status()
        pipe = joblib.load(io.BytesIO(r.content))
    else:
        st.error("Model not found. Upload models/pipeline.pkl or set MODEL_URL.")
        st.stop()

    num_cols, cat_cols = [], []
    try:
        pre = pipe.named_steps.get("preproc")
        # pre.transformers_ = [(name, transformer, column_names), ...]
        if getattr(pre, "transformers_", None):
            for name, _trans, cols in pre.transformers_:
                if isinstance(cols, list):
                    if name.startswith("num"):
                        num_cols.extend(cols)
                    elif name.startswith("cat"):
                        cat_cols.extend(cols)
    except Exception:
        pass

    return pipe, list(dict.fromkeys(num_cols)), list(dict.fromkeys(cat_cols))  # de-dup


def load_sample_df(expected_cols):
    """
    Loads a sample dataset to help with UI defaults:
    - ./data/hmeq.csv or CSV_URL (RAW GitHub)
    Returns a DataFrame (may be empty).
    """
    csv_local = "data/hmeq.csv"
    csv_url = os.getenv("CSV_URL")  # optional RAW GitHub URL
    df = None

    if os.path.isfile(csv_local):
        try:
            df = pd.read_csv(csv_local)
        except Exception:
            df = None

    if df is None and csv_url:
        try:
            r = requests.get(csv_url, timeout=30)
            if r.ok:
                df = pd.read_csv(io.BytesIO(r.content))
        except Exception:
            df = None

    if df is None:
        df = pd.DataFrame(columns=expected_cols)

    # Drop target if present
    df = df.drop(columns=["BAD"], errors="ignore")
    return df


def coerce_to_expected(df, expected_cols):
    """Ensure df has expected columns and order; create missing with NA."""
    if not expected_cols:
        return df
    out = df.copy()
    for c in expected_cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[expected_cols]


def score_df(pipe, df, threshold=0.5):
    """Return dataframe with probability and label using the given threshold."""
    proba = pipe.predict_proba(df)[:, 1]
    label = (proba >= threshold).astype(int)  # 1 = BAD
    out = df.copy()
    out["prob_BAD"] = proba
    out["label_BAD"] = label
    return out


# ---------------------------
# App state / UI helpers
# ---------------------------
pipe, num_cols, cat_cols = load_model()
expected_cols = (num_cols + cat_cols) if (num_cols or cat_cols) else []
sample_df = load_sample_df(expected_cols)

with st.sidebar:
    st.markdown("### Settings")
    thr = st.slider("Decision threshold (BAD=1)", 0.0, 1.0, 0.50, 0.01)
    st.caption("Adjust threshold according to your risk policy.")
    st.markdown("---")
    st.markdown("**Model source**")
    if os.path.isfile("models/pipeline.pkl"):
        st.caption("Loaded from: `models/pipeline.pkl`")
    elif os.getenv("MODEL_URL"):
        st.caption("Loaded from: `MODEL_URL`")
    else:
        st.caption("Model not found.")

tabs = st.tabs(["Real-time input", "CSV batch", "Model info"])

# ---------------------------
# Real-time single record
# ---------------------------
with tabs[0]:
    st.markdown("#### Enter a single case")
    if expected_cols:
        st.caption("Fields inferred from the training pipeline.")
    else:
        st.caption("No feature list found in the pipeline. Using sample CSV columns (if any).")

    # Determine fields to show: prefer pipeline-derived columns; else sample columns
    fields = expected_cols or sample_df.columns.tolist()

    # Build inputs with sensible defaults from sample_df
    defaults = {}
    if not sample_df.empty:
        # Use medians/mode as defaults
        for c in fields:
            if c in sample_df.columns:
                s = sample_df[c]
                if pd.api.types.is_numeric_dtype(s):
                    defaults[c] = float(s.dropna().median()) if s.notna().any() else 0.0
                else:
                    defaults[c] = str(s.dropna().mode().iloc[0]) if s.notna().any() else ""
            else:
                defaults[c] = 0.0 if c in num_cols else ""

    else:
        for c in fields:
            defaults[c] = 0.0 if (c in num_cols or c.lower().startswith(("amt","num","val"))) else ""

    # Two-column responsive layout
    left, right = st.columns(2)
    row = {}
    for i, c in enumerate(fields):
        target_col = left if i % 2 == 0 else right
        if c in num_cols or pd.api.types.is_numeric_dtype(sample_df[c]) if c in sample_df.columns else False:
            row[c] = target_col.number_input(c, value=float(defaults.get(c, 0.0)))
        else:
            # propose options from sample if available
            opts = []
            if c in sample_df.columns and not sample_df.empty:
                vals = sample_df[c].dropna().astype(str).unique().tolist()
                vals = vals[:50]  # keep UI lean
                opts = sorted(vals)
            if opts:
                row[c] = target_col.selectbox(c, opts, index=0)
            else:
                row[c] = target_col.text_input(c, value=str(defaults.get(c, "")))

    if st.button("Score case"):
        df_in = pd.DataFrame([row])
        df_in = coerce_to_expected(df_in, fields)
        scored = score_df(pipe, df_in, thr)
        proba = float(scored["prob_BAD"].iloc[0])
        label = int(scored["label_BAD"].iloc[0])
        st.success("Scored 1 record.")
        m1, m2 = st.columns(2)
        m1.metric("Probability of BAD", f"{proba:0.3f}")
        m2.metric("Predicted label", "BAD = 1" if label == 1 else "GOOD = 0")
        st.dataframe(scored)

# ---------------------------
# CSV batch
# ---------------------------
with tabs[1]:
    st.markdown("#### Upload a CSV")
    st.caption("Schema should match training features (no BAD column).")
    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            df = df.drop(columns=["BAD"], errors="ignore")
            df = coerce_to_expected(df, expected_cols or df.columns.tolist())
            scored = score_df(pipe, df, thr)
            st.success(f"Scored {len(scored)} rows.")
            st.dataframe(scored.head(50))
            csv_bytes = scored.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", data=csv_bytes, file_name="scored.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ---------------------------
# Model info
# ---------------------------
with tabs[2]:
    st.markdown("#### Model & Features")
    st.write(f"- Numeric features: {len(num_cols)}")
    st.write(f"- Categorical features: {len(cat_cols)}")
    if num_cols:
        with st.expander("Show numeric features"):
            st.code("\n".join(num_cols))
    if cat_cols:
        with st.expander("Show categorical features"):
            st.code("\n".join(cat_cols))
    # metadata.json (optional)
    meta_path = "models/metadata.json"
    meta_url = os.getenv("METADATA_URL")
    meta = None
    if os.path.isfile(meta_path):
        try:
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
        except Exception:
            meta = None
    elif meta_url:
        try:
            r = requests.get(meta_url, timeout=15)
            if r.ok:
                meta = r.json()
        except Exception:
            meta = None
    if meta:
        st.markdown("#### metadata.json")
        st.json(meta)
