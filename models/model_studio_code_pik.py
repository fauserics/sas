# === SAS Model Studio - Open Source Code (Python) ===
# Genera SIEMPRE: dm_scoreddf + artefactos en dm_nodedir (prefijo rpt_)

import os, json, pickle, traceback
from datetime import datetime
import pandas as pd
import numpy as np

# joblib opcional (si no está, usamos pickle)
try:
    import joblib
    def save_model(obj, path): joblib.dump(obj, path)
except Exception:
    def save_model(obj, path):
        with open(path, "wb") as f: pickle.dump(obj, f)

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ---------- util ----------
def ohe_dense_kwargs():
    try:
        OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        return {"sparse_output": False, "handle_unknown": "ignore"}
    except TypeError:
        return {"sparse": False, "handle_unknown": "ignore"}

def to_binary(y):
    y = pd.to_numeric(y, errors="coerce").fillna(0)
    return (y > 0).astype(int)

def safe_make_dir(p):
    os.makedirs(p, exist_ok=True)

def write_text(path, txt):
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

# ---------- ejecución protegida ----------
try:
    # 1) Datos del nodo
    X_all = dm_inputdf.loc[:, dm_input].copy()
    y_all = to_binary(dm_inputdf[dm_dec_target])

    # Partición
    if dm_partitionvar:
        mask_tr = dm_inputdf[dm_partitionvar] == dm_partition_train_val
    else:
        mask_tr = pd.Series(True, index=dm_inputdf.index)

    X_tr, y_tr = X_all.loc[mask_tr], y_all.loc[mask_tr]

    # 2) Preprocesamiento (num: imputación; cat: imputación + OHE denso)
    num_cols = X_tr.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_tr.columns if X_tr[c].dtype == "object"]

    preproc = ColumnTransformer([
        ("num", SimpleImputer(strategy="mean"), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(**ohe_dense_kwargs()))]), cat_cols)
    ])

    # 3) Modelo y validación robusta (reduce n_splits si hay pocas clases)
    min_class = int(min((y_tr==0).sum(), (y_tr==1).sum()))
    n_splits = max(2, min(5, min_class)) if len(y_tr) >= 2 else 2

    model = LogisticRegression(max_iter=1000, random_state=42)
    pipe = Pipeline([("preproc", preproc), ("clf", model)])

    try:
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            auc = float(cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean())
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")

    pipe.fit(X_tr, y_tr)

    # 4) Scoring para el nodo (OBLIGATORIO: mismas filas que dm_inputdf)
    proba = pipe.predict_proba(X_all)[:, 1]
    dm_scoreddf = pd.DataFrame(index=dm_inputdf.index)
    if isinstance(dm_predictionvar, (list, tuple)) and len(dm_predictionvar) >= 2:
        dm_scoreddf[dm_predictionvar[1]] = proba
        dm_scoreddf[dm_predictionvar[0]] = 1.0 - proba
    else:
        dm_scoreddf["P_1"] = proba
        dm_scoreddf["P_0"] = 1.0 - proba

    # 5) Artefactos en dm_nodedir (prefijo rpt_)
    outdir = dm_nodedir if "dm_nodedir" in globals() else os.getcwd()
    safe_make_dir(outdir)

    # Modelo
    model_path = os.path.join(outdir, "rpt_pipeline.pkl")
    save_model(pipe, model_path)

    # Metadata
    try:
        feat_names = pipe.named_steps["preproc"].get_feature_names_out().tolist()
    except Exception:
        feat_names = []
    meta = {
        "model": "LogisticRegression",
        "metrics_used": "AUC",
        "auc": (None if np.isnan(auc) else round(auc, 4)),
        "n_features_after_preprocess": (len(feat_names) if feat_names else None),
        "feature_names": (feat_names or None),
        "train_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ms_project_info": {
            "target": dm_dec_target,
            "inputs": dm_input,
            "partition_var": dm_partitionvar
        }
    }
    write_text(os.path.join(outdir, "rpt_metadata.json"), json.dumps(meta, indent=2, ensure_ascii=False))

    # Score script (batch local)
    score_py = """import sys, pandas as pd, joblib, pickle, os
def _load(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, 'rb') as f: return pickle.load(f)
pipe = _load(os.path.join('models','pipeline.pkl')) if os.path.isdir('models') else _load('rpt_pipeline.pkl')
def score_dataframe(df: pd.DataFrame): return pipe.predict_proba(df)[:,1]
if __name__ == '__main__':
    if len(sys.argv)<2:
        print('Usage: python score.py <in_csv> [out_csv]'); raise SystemExit(1)
    inp = sys.argv[1]; outp = sys.argv[2] if len(sys.argv)>2 else None
    df = pd.read_csv(inp); df['Score'] = score_dataframe(df)
    df.to_csv(outp, index=False) if outp else print(df.head())
"""
    write_text(os.path.join(outdir, "rpt_score.py"), score_py)

    # Sample CSV (sin target)
    dm_inputdf.loc[:, dm_input].head(500).to_csv(os.path.join(outdir, "rpt_sample.csv"), index=False)

    # README rápido
    readme = f"""Artifacts generated on {datetime.now():%Y-%m-%d %H:%M:%S}
- rpt_pipeline.pkl      (model)
- rpt_metadata.json     (metadata)
- rpt_score.py          (batch scoring helper)
- rpt_sample.csv        (sample inputs)
Descargá estos archivos desde Results y súbilos a tu repo (models/ y data/)."""
    write_text(os.path.join(outdir, "rpt_README.txt"), readme)

except Exception as e:
    # Log detallado y fallback de scoring para que el nodo NO falle
    err_path = os.path.join(dm_nodedir if "dm_nodedir" in globals() else os.getcwd(), "rpt_error.txt")
    write_text(err_path, "EXCEPTION:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)))
    try:
        dm_scoreddf = pd.DataFrame(index=dm_inputdf.index)
        if isinstance(dm_predictionvar, (list, tuple)) and len(dm_predictionvar) >= 2:
            dm_scoreddf[dm_predictionvar[1]] = 0.0
            dm_scoreddf[dm_predictionvar[0]] = 1.0
        else:
            dm_scoreddf["P_1"] = 0.0
            dm_scoreddf["P_0"] = 1.0
    except Exception:
        raise
