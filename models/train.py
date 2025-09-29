# train.py — universal (SAS / GitHub / local)
import os, sys, json, warnings, argparse, joblib
warnings.filterwarnings("ignore")

# ----------------- CLI / ENV -----------------
def get_args():
    p = argparse.ArgumentParser(description="Binary training (evento=1) con export a artefactos")
    p.add_argument("--data", choices=["cas", "csv"], default=os.getenv("DATA_MODE", "cas"))
    p.add_argument("--caslib", default=os.getenv("CAS_LIB", "Public"))
    p.add_argument("--castable", default=os.getenv("CAS_TABLE", "HMEQ"))
    p.add_argument("--target", default=os.getenv("TARGET", "BAD"))
    p.add_argument("--csv", default=os.getenv("CSV_PATH", ""))               # path local
    p.add_argument("--csv_url", default=os.getenv("CSV_URL", ""))            # URL (fallback)
    p.add_argument("--test_size", type=float, default=float(os.getenv("TEST_SIZE", "0.3")))
    p.add_argument("--random_state", type=int, default=int(os.getenv("RANDOM_STATE", "42")))
    p.add_argument("--metric", choices=["roc_auc", "pr_auc", "f1"], default=os.getenv("BEST_METRIC", "roc_auc"))
    p.add_argument("--outdir", default=os.getenv("OUT_DIR", os.getcwd()))
    return p.parse_args()

# ----------------- DATA LOADER -----------------
def load_from_cas(caslib, castable, target):
    try:
        import swat
    except Exception as e:
        raise RuntimeError("SWAT no disponible, no puedo usar CAS") from e
    # Conexión “amigable” para SAS Compute, con fallbacks si corre fuera
    try:
        s = swat.CAS(); s.serverstatus()
    except Exception:
        host = os.getenv("CAS_HOST") or os.getenv("SAS_CAS_HOST")
        port = int(os.getenv("CAS_PORT", "5570"))
        protocol = os.getenv("CAS_PROTOCOL", "http")
        token = os.getenv("SAS_SERVICES_TOKEN") or os.getenv("ACCESS_TOKEN")
        if token:
            s = swat.CAS(host, port, protocol=protocol, token=token)
        else:
            user = os.getenv("CAS_USER"); pwd = os.getenv("CAS_PASSWORD")
            s = swat.CAS(host, port, protocol=protocol, username=user, password=pwd)
    tbl = s.CASTable(castable, caslib=caslib)
    df = tbl.to_frame()
    if target not in df.columns:
        raise ValueError(f"Target '{target}' no está en {caslib}.{castable}")
    return df

def load_from_csv(path, url, target):
    import pandas as pd, requests, io
    if path and os.path.exists(path):
        df = pd.read_csv(path)
    elif url:
        r = requests.get(url, timeout=30); r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
    else:
        # fallback público HMEQ si no pasan fuente (dataset de muestra)
        default = "https://raw.githubusercontent.com/selva86/datasets/master/HMEQ.csv"
        r = requests.get(default, timeout=30); r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
    if target not in df.columns:
        raise ValueError(f"Target '{target}' no está en el CSV")
    return df

# ----------------- MODELOS / PIPELINE -----------------
def make_ohe():
    # Compatibilidad sklearn >=1.2 (sparse_output) vs <1.2 (sparse)
    from sklearn.preprocessing import OneHotEncoder
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def fit_and_select(df, target, test_size, random_state, metric_key):
    import numpy as np, pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 f1_score, precision_recall_curve, log_loss)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    y = df[target].astype(int).values
    X = df.drop(columns=[target])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline([("imp", SimpleImputer(strategy="median"))])
    categorical = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                            ("ohe", make_ohe())]) if cat_cols else None

    transformers = [("num", numeric, num_cols)]
    if cat_cols: transformers.append(("cat", categorical, cat_cols))
    pre = ColumnTransformer(transformers)

    candidates = {
        "logit": LogisticRegression(max_iter=1000, solver="liblinear",
                                    class_weight="balanced", random_state=random_state),
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                     class_weight="balanced", random_state=random_state),
        "gb": GradientBoostingClassifier(random_state=random_state)
    }

    # balance simple por sample_weight
    pos_w = (len(y)/(2.0*(y==1).sum())) if (y==1).sum()>0 else 1.0
    neg_w = (len(y)/(2.0*(y==0).sum())) if (y==0).sum()>0 else 1.0
    sw = np.where(y==1, pos_w, neg_w)

    Xtr, Xte, ytr, yte, sw_tr, sw_te = train_test_split(
        X, y, sw, test_size=test_size, stratify=y, random_state=random_state
    )

    results = []
    for name, est in candidates.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        try:
            pipe.fit(Xtr, ytr, model__sample_weight=sw_tr)
        except TypeError:
            pipe.fit(Xtr, ytr)
        p1 = pipe.predict_proba(Xte)[:, 1]
        roc = roc_auc_score(yte, p1)
        prc = average_precision_score(yte, p1)
        try:
            ll = log_loss(yte, np.c_[1 - p1, p1])
        except Exception:
            ll = float("nan")
        prec, rec, thr = precision_recall_curve(yte, p1)
        f1s = np.where((prec+rec)>0, 2*prec*rec/(prec+rec), 0.0)
        idx = int(np.nanargmax(f1s[:-1])) if len(f1s)>1 else 0
        best_thr = float(thr[idx]) if len(thr)>0 else 0.5
        f1 = f1_score(yte, (p1>=best_thr).astype(int), pos_label=1)
        results.append({
            "name": name, "auc": float(roc), "aupr": float(prc),
            "logloss": float(ll), "f1_best": float(f1),
            "thr": best_thr, "pipe": pipe
        })

    keymap = {"roc_auc":"auc", "pr_auc":"aupr", "f1":"f1_best"}
    key = keymap.get(metric_key, "auc")
    best = max(results, key=lambda r: r[key])
    return best, results, num_cols, cat_cols, X.columns.tolist()

# ----------------- ARTEFACTOS -----------------
def save_artifacts(best, results, target, num_cols, cat_cols, all_cols, outdir):
    import pandas as pd, numpy as np, textwrap, json, os, joblib
    os.makedirs(outdir, exist_ok=True)

    # pipeline
    joblib.dump(best["pipe"], os.path.join(outdir, "pipeline.pkl"))

    # metadata
    def levels_for(col):
        vals = pd.Series(col).dropna().astype(str).unique().tolist()
        return vals[:50] if len(vals) > 50 else vals

    metadata = {
        "model_name": f"hmeq_{best['name']}",
        "version": "v1",
        "target": target,
        "threshold": float(best["thr"]),
        "selection_metric": "roc_auc",
        "metrics_holdout": {
            "roc_auc": best["auc"], "pr_auc": best["aupr"], "f1_at_best": best["f1_best"]
        },
        "inputs": [
            {"name": c,
             "type": ("number" if c in num_cols else "string"),
             "required": False,
             **({"levels": []} if c in num_cols else {})}
            for c in all_cols
        ],
        "outputs": [{"name":"p_1","type":"number"}, {"name":"label","type":"int"}]
    }
    with open(os.path.join(outdir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # resumen de modelos
    models_trained = {
        "best_by": "roc_auc",
        "models": [{k: (float(v) if isinstance(v,(int,float)) else v)
                    for k,v in r.items() if k!="pipe"} for r in results]
    }
    with open(os.path.join(outdir, "models_trained.json"), "w", encoding="utf-8") as f:
        json.dump(models_trained, f, indent=2)

    # score.py
    score_py = """
import os, threading, pandas as pd, joblib, json
_LOCK=threading.Lock(); _MODEL=None; _THRESHOLD=None
def _meta_thr():
    here=os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    try:
        with open(os.path.join(here,"metadata.json"),"r",encoding="utf-8") as f:
            return float(json.load(f).get("threshold",0.5))
    except Exception:
        return 0.5
def _ensure_pkl(p):
    if os.path.exists(p): return
    url=os.environ.get("PIPELINE_URL")
    if url:
        import requests
        r=requests.get(url, timeout=20); r.raise_for_status()
        open(p,"wb").write(r.content)
def _load():
    global _MODEL,_THRESHOLD
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                here=os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
                pkl=os.path.join(here,"pipeline.pkl"); _ensure_pkl(pkl)
                _MODEL=joblib.load(pkl)
                th=os.environ.get("THRESHOLD")
                _THRESHOLD=float(th) if th is not None else _meta_thr()
    return _MODEL,_THRESHOLD
def _score_one(d:dict):
    m,t=_load(); X=pd.DataFrame([d]); p1=float(m.predict_proba(X)[0,1])
    return {"p_1":p1, "label": int(p1>=t)}
def score(record):
    if isinstance(record, list): return [_score_one(r) for r in record]
    return _score_one(record)
""".strip()
    with open(os.path.join(outdir, "score.py"), "w", encoding="utf-8") as f:
        f.write(score_py)

    # requirements.txt (para la app o CI)
    with open(os.path.join(outdir, "requirements.txt"), "w") as f:
        f.write("\n".join([
            "pandas>=2.0",
            "numpy>=1.24",
            "scikit-learn>=1.0",
            "joblib>=1.2",
            "requests>=2.31"
        ]) + "\n")

# ----------------- MAIN -----------------
def main():
    args = get_args()
    print("args:", vars(args))

    # cargar datos
    if args.data == "cas":
        try:
            df = load_from_cas(args.caslib, args.castable, args.target)
        except Exception as e:
            print("CAS no disponible, fallback a CSV:", e)
            df = load_from_csv(args.csv, args.csv_url, args.target)
    else:
        df = load_from_csv(args.csv, args.csv_url, args.target)

    # aseguramos target binario 0/1 y evento positivo = 1
    df[args.target] = df[args.target].astype(int)

    best, results, num_cols, cat_cols, all_cols = fit_and_select(
        df, args.target, args.test_size, args.random_state, args.metric
    )

    save_artifacts(best, results, args.target, num_cols, cat_cols, all_cols, args.outdir)
    print(f"OK — mejor modelo: {best['name']}  AUC={best['auc']:.4f}  thr*={best['thr']:.3f}")
    print(f"Artefactos en: {args.outdir}")

if __name__ == "__main__":
    main()
