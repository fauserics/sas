# === Model Studio - Open Source Code (Python) ===
# Requiere: Supervised learning = ON, Generate data frame = ON

import os, json, io, zipfile
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ---------------- utils ----------------
def _ohe_dense_kwargs():
    try:
        OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        return {"sparse_output": False, "handle_unknown": "ignore"}
    except TypeError:  # scikit-learn <=1.1
        return {"sparse": False, "handle_unknown": "ignore"}

def _build_preproc(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_tr = SimpleImputer(strategy="mean")
    cat_tr = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(**_ohe_dense_kwargs()))
    ])
    return ColumnTransformer([("num", num_tr, num_cols),
                              ("cat", cat_tr, cat_cols)]), num_cols, cat_cols

def _train_select(X, y, preproc):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results, best_name, best_pipe, best_auc = {}, None, None, -1.0
    for name, mdl in models.items():
        pipe = Pipeline([("preproc", preproc), ("clf", mdl)])
        auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
        results[name] = float(auc)
        if auc > best_auc:
            best_auc, best_name, best_pipe = auc, name, pipe
    best_pipe.fit(X, y)
    return best_pipe, best_name, results

def _to_binary_series(y):
    y = pd.to_numeric(y, errors="coerce")
    y = (y.fillna(0) > 0).astype(int)
    return y

# ---------------- tomar datos del nodo ----------------
# dm_inputdf, dm_input, dm_dec_target, dm_partitionvar, dm_partition_train_val,
# dm_predictionvar, dm_nodedir son provistos por Model Studio. :contentReference[oaicite:1]{index=1}

X_all = dm_inputdf.loc[:, dm_input].copy()
y_all = _to_binary_series(dm_inputdf[dm_dec_target])

if dm_partitionvar:
    mask_tr = dm_inputdf[dm_partitionvar] == dm_partition_train_val
else:
    mask_tr = pd.Series(True, index=dm_inputdf.index)

X_tr, y_tr = X_all.loc[mask_tr], y_all.loc[mask_tr]

# ---------------- entrenar ----------------
preproc, num_cols, cat_cols = _build_preproc(X_tr)
pipe, best_name, results = _train_select(X_tr, y_tr, preproc)

# ---------------- score para el nodo ----------------
proba = pipe.predict_proba(X_all)[:, 1]
dm_scoreddf = pd.DataFrame(index=dm_inputdf.index)  # ¡mismo # filas! :contentReference[oaicite:2]{index=2}
if isinstance(dm_predictionvar, (list, tuple)) and len(dm_predictionvar) >= 2:
    dm_scoreddf[dm_predictionvar[1]] = proba
    dm_scoreddf[dm_predictionvar[0]] = 1.0 - proba
else:
    dm_scoreddf["P_1"] = proba
    dm_scoreddf["P_0"] = 1.0 - proba

# ---------------- artefactos para GitHub/Streamlit ----------------
os.makedirs(dm_nodedir, exist_ok=True)
bundle_root = os.path.join(dm_nodedir, "streamlit_bundle")  # carpeta base del zip
paths = {
    "models": os.path.join(bundle_root, "models"),
    "data":   os.path.join(bundle_root, "data"),
}
for p in paths.values():
    os.makedirs(p, exist_ok=True)

# 1) modelo
model_path = os.path.join(paths["models"], "pipeline.pkl")
joblib.dump(pipe, model_path)

# 2) metadata
try:
    feat_names = pipe.named_steps["preproc"].get_feature_names_out().tolist()
except Exception:
    feat_names = []
meta = {
    "best_model": best_name,
    "metrics_used": "AUC",
    "metrics": {m: round(v, 4) for m, v in results.items()},
    "n_features_after_preprocess": (len(feat_names) if feat_names else None),
    "feature_names": feat_names or None,
    "model_params": pipe.named_steps["clf"].get_params(),
    "train_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "ms_project_info": {
        "target": dm_dec_target,
        "inputs": dm_input,
        "partition_var": dm_partitionvar,
    },
}
with open(os.path.join(paths["models"], "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

# 3) muestra CSV para la app (sin target)
sample = X_all.head(500).copy()
sample.to_csv(os.path.join(paths["data"], "sample.csv"), index=False)

# 4) score.py (batch local)
score_py = """import sys, pandas as pd, joblib
pipe = joblib.load("models/pipeline.pkl")
def score_dataframe(df: pd.DataFrame): return pipe.predict_proba(df)[:,1]
if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python score.py <in_csv> [out_csv]"); raise SystemExit(1)
    inp, outp = sys.argv[1], (sys.argv[2] if len(sys.argv)>2 else None)
    df = pd.read_csv(inp); df["Score"]=score_dataframe(df)
    (df.to_csv(outp, index=False) if outp else print(df.head()))"""
with open(os.path.join(bundle_root, "score.py"), "w", encoding="utf-8") as f:
    f.write(score_py + "\n")

# 5) app.py (Streamlit) — título pedido
app_py = """import os, io, json, requests
import pandas as pd
import streamlit as st
import joblib

APP_TITLE = "Real Time Scoring App (porwered by SAS)"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

def resolve_model_path():
    for p in ["models/pipeline.pkl", "../models/pipeline.pkl", "sas/models/pipeline.pkl"]:
        if os.path.isfile(p): return p
    return None

def load_model():
    local_path = resolve_model_path()
    model_url = os.getenv("MODEL_URL")
    if local_path:
        pipe = joblib.load(local_path); source = f"local: {local_path}"
    elif model_url:
        import requests
        r = requests.get(model_url, timeout=30); r.raise_for_status()
        pipe = joblib.load(io.BytesIO(r.content)); source = "MODEL_URL"
    else:
        st.error("Model not found. Place models/pipeline.pkl or set MODEL_URL."); st.stop()
    num_cols, cat_cols = [], []
    try:
        pre = pipe.named_steps.get("preproc")
        if getattr(pre, "transformers_", None):
            for name, _t, cols in pre.transformers_:
                if isinstance(cols, list):
                    (num_cols if name.startswith("num") else cat_cols).extend(cols)
    except Exception: pass
    # dedup
    num_cols = list(dict.fromkeys(num_cols)); cat_cols = list(dict.fromkeys(cat_cols))
    return pipe, num_cols, cat_cols, source

def coerce_to_expected(df, expected_cols):
    if not expected_cols: return df
    for c in expected_cols:
        if c not in df.columns: df[c] = pd.NA
    return df[expected_cols]

def score_df(pipe, df, threshold=0.5):
    proba = pipe.predict_proba(df)[:,1]
    label = (proba>=threshold).astype(int)
    out = df.copy(); out["prob_BAD"]=proba; out["label_BAD"]=label
    return out

pipe, num_cols, cat_cols, src = load_model()
expected = (num_cols+cat_cols) if (num_cols or cat_cols) else []
with st.sidebar:
    st.markdown("### Settings")
    thr = st.slider("Decision threshold (BAD=1)", 0.0, 1.0, 0.50, 0.01)
    st.caption(f"Model loaded from: {src}")

tabs = st.tabs(["Real-time input", "CSV batch", "Model info"])

# Real-time
with tabs[0]:
    st.markdown("#### Enter a single case")
    fields = expected or []
    # defaults from data/sample.csv if exists
    defaults, sample_df = {}, None
    for cand in ["data/sample.csv", "../data/sample.csv"]:
        if os.path.isfile(cand):
            sample_df = pd.read_csv(cand); break
    if sample_df is None: sample_df = pd.DataFrame(columns=fields)
    sample_df = sample_df.drop(columns=["BAD"], errors="ignore")
    if not sample_df.empty:
        for c in fields:
            if c in sample_df.columns:
                s = sample_df[c]
                defaults[c] = float(s.dropna().median()) if s.dtype.kind in "if" else str(s.dropna().mode().iloc[0]) if s.notna().any() else (0.0 if s.dtype.kind in "if" else "")
            else:
                defaults[c] = 0.0
    left, right = st.columns(2); row={}
    for i,c in enumerate(fields):
        tgt = left if i%2==0 else right
        if sample_df is not None and c in sample_df.columns and sample_df[c].dtype.kind in "if":
            row[c]=tgt.number_input(c, value=float(defaults.get(c,0.0)))
        else:
            opts = []
            if sample_df is not None and c in sample_df.columns:
                vals = sample_df[c].dropna().astype(str).unique().tolist()[:50]; opts = sorted(vals)
            row[c] = (tgt.selectbox(c, opts, index=0) if opts else tgt.text_input(c, value=str(defaults.get(c,""))))
    if st.button("Score case"):
        df_in = pd.DataFrame([row]); df_in = coerce_to_expected(df_in, fields)
        scored = score_df(pipe, df_in, thr)
        proba=float(scored["prob_BAD"].iloc[0]); label=int(scored["label_BAD"].iloc[0])
        st.success("Scored 1 record.")
        c1,c2 = st.columns(2)
        c1.metric("Probability of BAD", f"{proba:0.3f}")
        c2.metric("Predicted label", "BAD = 1" if label==1 else "GOOD = 0")
        st.dataframe(scored)

# CSV batch
with tabs[1]:
    st.markdown("#### Upload a CSV")
    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        df = df.drop(columns=["BAD"], errors="ignore")
        df = coerce_to_expected(df, expected or df.columns.tolist())
        scored = score_df(pipe, df, thr)
        st.success(f"Scored {len(scored)} rows.")
        st.dataframe(scored.head(50))
        st.download_button("Download results CSV", data=scored.to_csv(index=False).encode("utf-8"),
                           file_name="scored.csv", mime="text/csv")

# Model info
with tabs[2]:
    st.markdown("#### Model & Features")
    st.write(f"- Numeric features: {len(num_cols)}"); st.write(f"- Categorical features: {len(cat_cols)}")
    if num_cols: st.expander("Show numeric features").write("\\n".join(num_cols))
    if cat_cols: st.expander("Show categorical features").write("\\n".join(cat_cols))
"""
with open(os.path.join(bundle_root, "app.py"), "w", encoding="utf-8") as f:
    f.write(app_py + "\n")

# 6) requirements para la app
reqs = """streamlit>=1.31
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
joblib>=1.3
requests>=2.31
"""
with open(os.path.join(bundle_root, "requirements.txt"), "w", encoding="utf-8") as f:
    f.write(reqs)

# 7) README (corto)
readme = f"""# Streamlit Scoring Bundle

This bundle was generated from SAS Model Studio (Open Source Code node) on {datetime.now():%Y-%m-%d %H:%M:%S}.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
