# === Model Studio - Open Source Code (Python) ===
import os, json, traceback
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

def _ohe_dense_kwargs():
    try:
        OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        return {"sparse_output": False, "handle_unknown": "ignore"}
    except TypeError:
        return {"sparse": False, "handle_unknown": "ignore"}

def _prep(X: pd.DataFrame):
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in X.columns if X[c].dtype == "object"]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="mean"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(**_ohe_dense_kwargs()))]), cat)
    ])
    return pre, num, cat

def _to_bin(y):
    y = pd.to_numeric(y, errors="coerce").fillna(0)
    return (y > 0).astype(int)

def _save_artifacts(pipe, auc, num, cat):
    os.makedirs(dm_nodedir, exist_ok=True)
    # modelo
    joblib.dump(pipe, os.path.join(dm_nodedir, "rpt_pipeline.pkl"))
    # metadata
    try:
        feats = pipe.named_steps["preproc"].get_feature_names_out().tolist()
    except Exception:
        feats = []
    meta = {
        "best_model": "LogisticRegression",
        "metrics_used": "AUC",
        "metrics": {"LogisticRegression": round(float(auc), 4)},
        "n_features_after_preprocess": (len(feats) if feats else None),
        "feature_names": feats or None,
        "train_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ms_project_info": {"target": dm_dec_target, "inputs": dm_input, "partition_var": dm_partitionvar}
    }
    with open(os.path.join(dm_nodedir, "rpt_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    # muestra CSV (sin target)
    dm_inputdf.loc[:, dm_input].head(500).to_csv(os.path.join(dm_nodedir, "rpt_sample.csv"), index=False)

try:
    # 1) datos del nodo
    X_all = dm_inputdf.loc[:, dm_input].copy()
    y_all = _to_bin(dm_inputdf[dm_dec_target])
    if dm_partitionvar:
        mask_tr = dm_inputdf[dm_partitionvar] == dm_partition_train_val
    else:
        mask_tr = pd.Series(True, index=dm_inputdf.index)
    X_tr, y_tr = X_all.loc[mask_tr], y_all.loc[mask_tr]

    # 2) prepro + modelo
    pre, num, cat = _prep(X_tr)
    pipe = Pipeline([("preproc", pre), ("clf", LogisticRegression(max_iter=1000, random_state=42))])

    # 3) AUC CV y fit final
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
    pipe.fit(X_tr, y_tr)

    # 4) scoring para el nodo (obligatorio)
    proba = pipe.predict_proba(X_all)[:, 1]
    dm_scoreddf = pd.DataFrame(index=dm_inputdf.index)
    if isinstance(dm_predictionvar, (list, tuple)) and len(dm_predictionvar) >= 2:
        dm_scoreddf[dm_predictionvar[1]] = proba
        dm_scoreddf[dm_predictionvar[0]] = 1.0 - proba
    else:
        dm_scoreddf["P_1"] = proba
        dm_scoreddf["P_0"] = 1.0 - proba

    # 5) artefactos para GitHub
    _save_artifacts(pipe, auc, num, cat)

except Exception as e:
    # Log del error y fallback para que el nodo no falle
    with open(os.path.join(dm_nodedir, "rpt_error.txt"), "w", encoding="utf-8") as f:
        f.write("EXCEPTION:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)))
    dm_scoreddf = pd.DataFrame(index=dm_inputdf.index)
    if isinstance(dm_predictionvar, (list, tuple)) and len(dm_predictionvar) >= 2:
        dm_scoreddf[dm_predictionvar[1]] = 0.0
        dm_scoreddf[dm_predictionvar[0]] = 1.0
    else:
        dm_scoreddf["P_1"] = 0.0
        dm_scoreddf["P_0"] = 1.0
