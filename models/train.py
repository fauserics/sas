#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import pandas as pd
import joblib
import tempfile

# SWAT opcional para CAS
try:
    import swat
    swat_available = True
except ImportError:
    swat_available = False


# --------------------------
# Utils
# --------------------------
def _onehot_dense_kwargs():
    """Forzar salida densa en OneHotEncoder según versión de scikit-learn."""
    try:
        from sklearn.preprocessing import OneHotEncoder
        _ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        return {"sparse_output": False, "handle_unknown": "ignore"}
    except TypeError:
        return {"sparse": False, "handle_unknown": "ignore"}


def _probe_work_paths():
    """
    Genera candidatos basados en WORK/TMP y un escaneo superficial (1–2 niveles).
    Útil cuando el CSV está en &WORK/hmeq.csv y PROC PYTHON tiene cwd distinto.
    """
    cands = []

    # 1) Env vars que podemos setear desde SAS
    for key in ("SAS_WORK", "SAS_WORK_PATH", "WORK", "TMPDIR"):
        wk = os.getenv(key)
        if wk and os.path.isdir(wk):
            cands.append(os.path.join(wk, "hmeq.csv"))
            # un nivel dentro
            try:
                for d in os.listdir(wk):
                    dp = os.path.join(wk, d)
                    if os.path.isdir(dp):
                        cands.append(os.path.join(dp, "hmeq.csv"))
            except Exception:
                pass

    # 2) tempdir del sistema
    tdir = tempfile.gettempdir()
    if tdir and os.path.isdir(tdir):
        cands.append(os.path.join(tdir, "hmeq.csv"))
        try:
            for d in os.listdir(tdir):
                dp = os.path.join(tdir, d)
                if os.path.isdir(dp):
                    cands.append(os.path.join(dp, "hmeq.csv"))
        except Exception:
            pass

    # 3) raíz típica de SAS Work en Linux (si existe)
    for root in ("/saswork", "/var/tmp"):
        if os.path.isdir(root):
            cands.append(os.path.join(root, "hmeq.csv"))
            # dos niveles como máximo
            try:
                for d in os.listdir(root):
                    dp = os.path.join(root, d)
                    if os.path.isdir(dp):
                        cands.append(os.path.join(dp, "hmeq.csv"))
                        for d2 in os.listdir(dp):
                            dp2 = os.path.join(dp, d2)
                            if os.path.isdir(dp2):
                                cands.append(os.path.join(dp2, "hmeq.csv"))
            except Exception:
                pass

    return cands


def find_csv_path():
    """
    Busca el dataset en este orden:
      1) argumento 1 (archivo o URL http/https)
      2) env HMEQ_CSV (archivo o URL http/https)
      3) ./hmeq.csv
      4) ./data/hmeq.csv
      5) &WORK/hmeq.csv (vía envs/TMP, con búsqueda 1–2 niveles)
    """
    cands = []
    if len(sys.argv) > 1:
        cands.append(sys.argv[1])
    if os.getenv("HMEQ_CSV"):
        cands.append(os.getenv("HMEQ_CSV"))
    cands += ["hmeq.csv", os.path.join("data", "hmeq.csv")]
    cands += _probe_work_paths()

    for p in cands:
        if not p:
            continue
        if isinstance(p, str) and p.lower().startswith(("http://", "https://")):
            return p  # pandas puede leer URLs
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        "No se encontró hmeq.csv. Pasá path/URL como arg, seteá HMEQ_CSV, "
        "o dejá el archivo en el cwd, ./data o &WORK."
    )


# --------------------------
# Data
# --------------------------
def load_data():
    """Carga HMEQ desde CAS (si hay credenciales) o CSV/URL."""
    use_cas = swat_available and os.getenv("CAS_USERNAME") and os.getenv("CAS_PASSWORD")

    if use_cas:
        conn = None
        try:
            host = os.getenv("CAS_HOST", "localhost")
            port = int(os.getenv("CAS_PORT", "5570"))
            protocol = os.getenv("CAS_PROTOCOL", "http")  # "http" o "cas"
            user = os.getenv("CAS_USERNAME")
            pwd = os.getenv("CAS_PASSWORD")

            conn = swat.CAS(host, port, user, pwd, protocol=protocol)
            castbl = conn.CASTable("HMEQ", caslib="Public")
            df = castbl.to_frame()
            print(f"[INFO] CAS Public.HMEQ -> {len(df)} filas", flush=True)
            try:
                conn.close()
            except Exception:
                pass
            return df
        except Exception as e:
            print("[WARN] CAS no disponible o credenciales inválidas. Se usará CSV. Detalle:", e, flush=True)
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

    # CSV/URL
    path = find_csv_path()
    df = pd.read_csv(path)
    print(f"[INFO] CSV '{path}' -> {len(df)} filas", flush=True)
    return df


def prepare_data(df):
    """Separa X,y; normaliza 'BAD' a binario; tipifica columnas."""
    df = df.rename(columns={c: c.upper() for c in df.columns})
    if "BAD" not in df.columns:
        raise KeyError("La columna 'BAD' no está en los datos.")

    y = df["BAD"].copy()
    if not set(pd.Series(y).dropna().unique()).issubset({0, 1}):
        y = (pd.to_numeric(y, errors="coerce").fillna(0) > 0).astype(int)
    else:
        y = y.astype(int)

    X = df.drop(columns=["BAD"]).copy()

    for col in X.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype(str)

    return X, y


def build_preprocessor(X):
    """ColumnTransformer (imputación + OHE denso)."""
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    num_cols = X.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_tr = SimpleImputer(strategy="mean")
    cat_tr = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**_onehot_dense_kwargs()))
    ])

    preproc = ColumnTransformer([
        ("num", num_tr, num_cols),
        ("cat", cat_tr, cat_cols),
    ])
    return preproc


# --------------------------
# Train & select
# --------------------------
def train_and_select_model(X, y, preprocessor):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        pipe = Pipeline([("preproc", preprocessor), ("clf", model)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        mean_auc, std_auc = scores.mean(), scores.std()
        results[name] = float(mean_auc)
        print(f"{name}: AUC = {mean_auc:.4f} (±{std_auc:.4f})", flush=True)

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    print(f"[INFO] Mejor: {best_model_name} (AUC={results[best_model_name]:.4f})", flush=True)

    from sklearn.pipeline import Pipeline
    best_pipe = Pipeline([("preproc", preprocessor), ("clf", best_model)])
    best_pipe.fit(X, y)
    return best_pipe, best_model_name, results


# --------------------------
# Artifacts
# --------------------------
def save_artifacts(pipeline, best_model_name, results):
    joblib.dump(pipeline, "pipeline.pkl")

    # Nombre de features (si está disponible)
    try:
        preproc = pipeline.named_steps["preproc"]
        try:
            feat_names = preproc.get_feature_names_out().tolist()
        except Exception:
            feat_names = []
    except Exception:
        feat_names = []

    meta = {
        "best_model": best_model_name,
        "metrics_used": "AUC",
        "metrics": {m: round(v, 4) for m, v in results.items()},
        "n_features_after_preprocess": len(feat_names) if feat_names else None,
        "feature_names": feat_names if feat_names else None,
        "model_params": pipeline.named_steps["clf"].get_params()
    }
    from datetime import datetime
    meta["train_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    score_py = r'''import sys
import pandas as pd
import joblib

pipeline = joblib.load("pipeline.pkl")

def score_dataframe(df: pd.DataFrame):
    return pipeline.predict_proba(df)[:, 1]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python score.py <archivo_csv_entrada> [archivo_csv_salida]")
        sys.exit(1)
    inp = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) > 2 else None
    data = pd.read_csv(inp)
    data["Score"] = score_dataframe(data)
    if outp:
        data.to_csv(outp, index=False)
        print(f"Resultados guardados en {outp}")
    else:
        print(data.head())
'''
    with open("score.py", "w", encoding="utf-8") as f:
        f.write(score_py)

    reqs = ["pandas", "numpy", "scikit-learn", "joblib"]
    if swat_available:
        reqs.append("swat")
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(reqs) + "\n")

    with open("metrics.csv", "w", encoding="utf-8") as f:
        f.write("Model,AUC\n")
        for m, auc in results.items():
            f.write(f"{m},{auc:.4f}\n")
        f.write(f"BestModel,{best_model_name}\n")

    metrics = {
        "model_auc": {m: round(v, 4) for m, v in results.items()},
        "best_model": best_model_name,
        "best_model_auc": round(results[best_model_name], 4)
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    df = load_data()
    X, y = prepare_data(df)
    preproc = build_preprocessor(X)
    pipe, best_name, res = train_and_select_model(X, y, preproc)
    save_artifacts(pipe, best_name, res)
    print("[OK] Entrenamiento completado. Artefactos: pipeline.pkl, metadata.json, metrics.(csv/json), score.py", flush=True)
