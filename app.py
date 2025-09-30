import os, io, json, requests
import pandas as pd
import streamlit as st

# ---- try joblib robustly ----
try:
    import joblib
except ModuleNotFoundError:
    try:
        # fallback legacy (scikit-learn <0.23) — probablemente no esté
        from sklearn.externals import joblib as joblib  # type: ignore
    except Exception:
        st.set_page_config(page_title="Real Time Scoring App", layout="wide")
        st.error("`joblib` is not installed. Add `joblib` to requirements.txt at the repo root and redeploy.")
        st.stop()

APP_TITLE = "Real Time Scoring App (powered by SAS)"  
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# ---------------------------
# Load model / metadata
# ---------------------------
def resolve_model_path():
    # app is under sas/app.py -> try multiple locations
    candidates = [
        "models/pipeline.pkl",          # if running from repo root
        "../models/pipeline.pkl",       # if app is in ./sas/
        "sas/models/pipeline.pkl",      # another common layout
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def load_model():
    """
    Loads a scikit-learn pipeline from local paths or MODEL_URL (RAW GitHub).
    Returns: (pipeline, numeric_cols, categorical_cols)
    """
    local_path = resolve_model_path()
    model_url = os.getenv("MODEL_URL")  # e.g., RAW GitHub URL to pipeline.pkl

    pipe = None
    if local_path:
        pipe = joblib.load(local_path)
        source = f"local: {local_path}"
    elif model_url:
        r = requests.get(model_url, timeout=30)
        r.raise_for_status()
        pipe = joblib.load(io.BytesIO(r.content))
        source = "MODEL_URL"
    else:
        st.error("Model not found. Place `models/pipeline.pkl` (relative to the app) or set `MODEL_URL` env var.")
        st.stop()

    num_cols, cat_cols = [], []
    try:
        pre = pipe.named_steps.get("preproc")
        if getattr(pre, "transformers_", None):
            for name, _trans, cols in pre.transformers_:
                if isinstance(cols, list):
                    if name.startswith("num"):
                        num_cols.extend(cols)
                    elif name.startswith("cat"):
                        cat_cols.extend(cols)
    except Exception:
        pass

    # dedup while preserving order
    num_cols = list(dict.fromkeys(num_cols))
    cat_cols = list(dict.fromkeys(cat_cols))
    return pipe, num_cols, cat_cols, source

def load_sample_df(expected_cols):
    """
    Load a sample CSV to populate defaults:
    - ./data/hmeq.csv
    - ../data/hmeq.csv
    - `CSV_URL` (RAW GitHub) if set
    """
    local_candidates = ["data/hmeq.csv", "../data/hmeq.csv", "sas/data/hmeq.csv"]
    df = None
    for p in local_candidates:
        if os.path.isfile(p):
            try:
                df = pd.read_csv(p)
                break
            except Exception:
                pass

    if df is None and os.getenv("CSV_URL"):
        try:
            r = requests.get(os.getenv("CSV_URL"), timeout=30)
            if r.ok:
                df = pd.read_csv(io.BytesIO(r.content))
        except Exception:
            df = None

    if df is None:
        df = pd.DataFrame(columns=expected_cols)

    df = df.drop(columns=["BAD"], errors="ignore")
    return df

def coerce_to_expected(df, expected_cols):
    if not expected_cols:
        return df
    out = df.copy()
    for c in expected_cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[expected_cols]

def score_df(pipe, df, threshold=0.5):
    proba = pipe.predict_proba(df)[:, 1]
    label = (proba >= threshold).astype(int)  # 1 = BAD
    out = df.copy()
    out["prob_BAD"] = proba
    out["label_BAD"] = label
    return out

# ---------------------------
# App state / UI
# ---------------------------
pipe, num_cols, cat_cols, model_source = load_model()
expected_cols = (num_cols + cat_cols) if (num_cols or cat_cols) else []
sample_df = load_sample_df(expected_cols)

with st.sidebar:
    st.markdown("### Settings")
    thr = st.slider("Decision threshold (BAD=1)", 0.0, 1.0, 0.50, 0.01)
    st.caption("Adjust threshold according to your risk policy.")
    st.markdown("---")
    st.caption(f"Model loaded from: {model_source}")

tabs = st.tabs(["Real-time input", "CSV batch", "Model info"])

# ---- Real-time input ----
with tabs[0]:
    st.markdown("#### Enter a single case")
    fields = expected_cols or sample_df.columns.tolist()

    # defaults from sample
    defaults = {}
    if not sample_df.empty:
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

    left, right = st.columns(2)
    row = {}
    for i, c in enumerate(fields):
        target_col = left if i % 2 == 0 else right
        if c in num_cols or (c in sample_df.columns and pd.api.types.is_numeric_dtype(sample_df[c])):
            row[c] = target_col.number_input(c, value=float(defaults.get(c, 0.0)))
        else:
            opts = []
            if c in sample_df.columns and not sample_df.empty:
                vals = sample_df[c].dropna().astype(str).unique().tolist()
                opts = sorted(vals[:50])
            row[c] = (target_col.selectbox(c, opts, index=0) if opts
                      else target_col.text_input(c, value=str(defaults.get(c, ""))))

    if st.button("Score case"):
        df_in = pd.DataFrame([row])
        df_in = coerce_to_expected(df_in, fields)
        scored = score_df(pipe, df_in, thr)
        proba = float(scored["prob_BAD"].iloc[0])
        label = int(scored["label_BAD"].iloc[0])
        st.success("Scored 1 record.")
        c1, c2 = st.columns(2)
        c1.metric("Probability of BAD", f"{proba:0.3f}")
        c2.metric("Predicted label", "BAD = 1" if label == 1 else "GOOD = 0")
        st.dataframe(scored)

# ---- CSV batch ----
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
            st.download_button(
                "Download results CSV",
                data=scored.to_csv(index=False).encode("utf-8"),
                file_name="scored.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ---- Model info ----
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
