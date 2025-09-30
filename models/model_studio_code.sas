#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, tempfile, warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import joblib

# SWAT opcional para CAS (solo modo standalone)
try:
    import swat
    swat_available = True
except Exception:
    swat_available = False


# ========= util comunes =========
def _ohe_dense_kwargs():
    """Forzar salida densa en OneHotEncoder según versión scikit-learn."""
    try:
        from sklearn.preprocessing import OneHotEncoder
        _ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        return {"sparse_output": False, "handle_unknown": "ignore"}
    except TypeError:
        return {"sparse": False, "handle_unknown": "ignore"}


def _probe_work_paths():
    """Candidatos típicos para hmeq.csv cuando SAS lo deja en WORK/TMP."""
    c = []
    for key in ("SAS_WORK", "SAS_WORK_PATH", "WORK", "TMPDIR"):
        wk = os.getenv(key)
        if wk and os.path.isdir(wk):
            c.append(os.path.join(wk, "hmeq.csv"))
            try:
                for d in os.listdir(wk):
                    dp = os.path.join(wk, d)
                    if os.path.isdir(dp):
                        c.append(os.path.join(dp, "hmeq.csv"))
            except Exception:
                pass
    # rutas comunes
    for root in (tempfile.gettempdir(), "/saswork", "/sasuser"):
        if root and os.path.isdir(root):
            c.append(os.path.join(root, "hmeq.csv"))
    # path directo muy usado
    c.append("/sasuser/hmeq.csv")
    return c


def find_csv_path():
    """
    Orden de búsqueda:
      1) argv[1] (archivo o URL http/https)
      2) env HMEQ_CSV (archivo o URL)
      3) ./hmeq.csv
      4) ./data/hmeq.csv
      5) WORK/TMP (/saswork, /sasuser, tempdir)
    """
    cands = []
    if len(sys.argv) > 1: cands.append(sys.argv[1])
    if os.getenv("HMEQ_CSV"): cands.append(os.getenv("HMEQ_CSV"))
    cands += ["hmeq.csv", os.path.join("data", "hmeq.csv")]
    cands += _probe_work_paths()

    for p in cands:
        if not p: 
            continue
        if str(p).lower().startswith(("http://", "https://")):
            return p
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("No se encontró hmeq.csv. Pasá path/URL como arg, seteá HMEQ_CSV, o dejalo en ./, ./data, &WORK, /sasuser.")


def prepare_xy(df: pd.DataFrame, target="BAD"):
    df = df.rename(columns={c: c.upper() for c in df.columns})
    if target.upper() not in df.columns:
        raise KeyError(f"No está la columna objetivo '{target}'.")
    y = df[target.upper()]
    if not set(pd.Series(y).dropna().unique()).issubset({0, 1}):
        y = (pd.to_numeric(y, errors="coerce").fillna(0) > 0).astype(int)
    else:
        y = y.astype(int)
    X = df.drop(columns=[target.upper()]).copy()
    # tipificados
    for col in X.select_dtypes(include=["float64","int64","float32","int32"]).columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype(str)
    return X, y


def build_preprocessor(X: pd.DataFrame):
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]

    num_tr = SimpleImputer(strategy="mean")
    cat_tr = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(**_ohe_dense_kwargs()))
    ])
    return ColumnTransformer([("num", num_tr, num_cols),
                              ("cat", cat_tr, cat_cols)])


def train_select(X, y, preproc):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline

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


# ========= modo Model Studio (Open Source Code) =========
def run_model_studio():
    # Variables provistas por el nodo (Generate data frame + Supervised ON)
    X_all = dm_inputdf.loc[:, dm_input]
    y_all = dm_inputdf[dm_dec_target]

    if dm_partitionvar:
        import numpy as np
        mask_tr = dm_inputdf[dm_partitionvar] == dm_partition_train_val
    else:
        mask_tr = pd.Series(True, index=dm_inputdf.index)

    X_tr, y_tr = X_all.loc[mask_tr], y_all.loc[mask_tr]

    preproc = build_preprocessor(X_tr)
    pipe, best_name, results = train_select(X_tr, y_tr, preproc)

    # Probabilidades para todas las filas del nodo
    proba = pipe.predict_proba(X_all)[:, 1]

    # Armar dm_scoreddf EXACTAMENTE con las columnas que pide el nodo
    # (dm_predictionvar suele ser [P_TARGET0, P_TARGET1])
    global dm_scoreddf
    dm_scoreddf = pd.DataFrame(index=dm_inputdf.index)
    if isinstance(dm_predictionvar, (list, tuple)) and len(dm_predictionvar) >= 2:
        dm_scoreddf[dm_predictionvar[1]] = proba
        dm_scoreddf[dm_predictionvar[0]] = 1.0 - proba
    else:
        # fallback razonable
        dm_scoreddf["P_1"] = proba
        dm_scoreddf["P_0"] = 1.0 - proba

    # Guardar artefactos/reporte en el dir del nodo (visible en Results)
    joblib.dump(pipe, os.path.join(dm_nodedir, "pipeline.pkl"))
    with open(os.path.join(dm_nodedir, "rpt_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "auc": results}, f, indent=2, ensure_ascii=False)


# ========= modo SAS Studio / GitHub (standalone) =========
def run_standalone():
    # 1) CAS opcional (solo si hay credenciales)
    use_cas = swat_available and os.getenv("CAS_USERNAME") and os.getenv("CAS_PASSWORD")
    df = None
    if use_cas:
        conn = None
        try:
            host = os.getenv("CAS_HOST", "localhost")
            port = int(os.getenv("CAS_PORT", "5570"))
            protocol = os.getenv("CAS_PROTOCOL", "http")
            user, pwd = os.getenv("CAS_USERNAME"), os.getenv("CAS_PASSWORD")
            conn = swat.CAS(host, port, user, pwd, protocol=protocol)
            df = conn.CASTable("HMEQ", caslib="Public").to_frame()
            print(f"[INFO] CAS Public.HMEQ -> {len(df)} filas", flush=True)
        except Exception as e:
            print("[WARN] CAS no disponible. Se usará CSV. Detalle:", e, flush=True)
        finally:
            try:
                if conn: conn.close()
            except Exception:
                pass

    # 2) CSV/URL (si no hubo CAS)
    if df is None:
        path = find_csv_path()
        df = pd.read_csv(path)
        print(f"[INFO] CSV '{path}' -> {len(df)} filas", flush=True)

    # 3) Entrenar y guardar artefactos en cwd
    X, y = prepare_xy(df, target="BAD")
    preproc = build_preprocessor(X)
    pipe, best_name, results = train_select(X, y, preproc)

    joblib.dump(pipe, "pipeline.pkl")
    # meta/metrics
    try:
        feat_names = pipe.named_steps["preproc"].get_feature_names_out().tolist()
    except Exception:
        feat_names = []
    meta = {
        "best_model": best_name,
        "metrics_used": "AUC",
        "metrics": {m: round(v, 4) for m, v in results.items()},
        "n_features_after_preprocess": len(feat_names) if feat_names else None,
        "feature_names": feat_names if feat_names else None,
        "model_params": pipe.named_steps["clf"].get_params()
    }
    from datetime import datetime
    meta["train_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    # score helper
    score_py = r'''import sys, pandas as pd, joblib
pipe = joblib.load("pipeline.pkl")
def score_dataframe(df: pd.DataFrame): return pipe.predict_proba(df)[:,1]
if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Uso: python score.py <in_csv> [out_csv]"); raise SystemExit(1)
    inp, outp = sys.argv[1], (sys.argv[2] if len(sys.argv)>2 else None)
    data = pd.read_csv(inp); data["Score"]=score_dataframe(data)
    (data.to_csv(outp, index=False) if outp else print(data.head()))'''
    with open("score.py", "w", encoding="utf-8") as f: f.write(score_py+"\n")

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_auc": {m: round(v,4) for m,v in results.items()},
            "best_model": best_name,
            "best_model_auc": round(results[best_name],4)
        }, f, indent=4)

    with open("metrics.csv", "w", encoding="utf-8") as f:
        f.write("Model,AUC\n")
        for m, auc in results.items(): f.write(f"{m},{auc:.4f}\n")
        f.write(f"BestModel,{best_name}\n")

    print("[OK] Entrenamiento completado (standalone). Artefactos: pipeline.pkl, metadata.json, metrics.(csv/json), score.py", flush=True)


# ========= entrypoint =========
if __name__ == "__main__":
    try:
        if "dm_inputdf" in globals():         # Model Studio
            run_model_studio()
        else:                                  # SAS Studio / GitHub
            run_standalone()
    except SystemExit:
        raise
    except Exception as e:
        # En Model Studio, dejar rastro visible en resultados del nodo
        if "dm_nodedir" in globals():
            try:
                with open(os.path.join(dm_nodedir, "rpt_error.txt"), "w", encoding="utf-8") as f:
                    f.write(repr(e))
            except Exception:
                pass
        raise
