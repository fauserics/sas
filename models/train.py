# train.py — compatible con SAS Studio y GitHub/local
import os, json, warnings
warnings.filterwarnings("ignore")

# ===== Config =====
DATA_MODE = os.getenv("DATA_MODE", "cas")     # "cas" o "csv"
CAS_LIB   = os.getenv("CAS_LIB", "Public")
CAS_TABLE = os.getenv("CAS_TABLE", "HMEQ")
CSV_PATH  = os.getenv("CSV_PATH", "")
CSV_URL   = os.getenv("CSV_URL", "")
TARGET    = os.getenv("TARGET", "BAD")        # evento positivo = 1

HERE    = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.getenv("OUT_DIR", HERE)

# Guardado modelo: joblib si está, sino pickle
try:
    import joblib as _joblib
except Exception:
    _joblib = None
import pickle as _pickle

def _dump_model(model, path):
    if _joblib is not None:
        _joblib.dump(model, path)
    else:
        with open(path, "wb") as f:
            _pickle.dump(model, f, protocol=_pickle.HIGHEST_PROTOCOL)

# ===== Util: CSV sin 'requests' =====
def _read_csv_from_url(url):
    from urllib.request import urlopen, Request
    import pandas as pd, io
    req = Request(url, headers={"User-Agent": "Python"})
    with urlopen(req, timeout=30) as r:
        data = r.read()
    return pd.read_csv(io.BytesIO(data))

# ===== Carga de datos =====
def load_from_cas():
    try:
        import swat
    except Exception as e:
        raise RuntimeError("SWAT no disponible; usa DATA_MODE=csv o instala swat") from e
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
    df = s.CASTable(CAS_TABLE, caslib=CAS_LIB).to_frame()
    if TARGET not in df.columns:
        raise ValueError("Target '%s' no está en %s.%s" % (TARGET, CAS_LIB, CAS_TABLE))
    return df

def load_from_csv():
    import pandas as pd
    if CSV_PATH and os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    url = CSV_URL or "https://raw.githubusercontent.com/selva86/datasets/master/HMEQ.csv"
    return _read_csv_from_url(url)

# ===== Prepro y modelos =====
def _make_ohe():
    from sklearn.preprocessing import OneHotEncoder
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn <1.2

def build_preprocessor(X):
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    numeric = Pipeline([("imp", SimpleImputer(strategy="median"))])
    transformers = []
    if num_cols:
        transformers.append(("num", numeric, num_cols))
    if cat_cols:
        categorical = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                ("ohe", _make_ohe())])
        transformers.append(("cat", categorical, cat_cols))
    from sklearn.compose import ColumnTransformer as CT
    if not transformers:
        transformers = [("num", numeric, [])]
    return CT(transformers), num_cols, cat_cols

def train_and_eval(df):
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 precision_recall_curve, roc_curve,
                                 accuracy_score, precision_score,
                                 recall_score, f1_score, confusion_matrix)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    y = df[TARGET].astype(int).values
    X = df.drop(columns=[TARGET])
    pre, num_cols, cat_cols = build_preprocessor(X)

    candidates = {
        "logit": LogisticRegression(max_iter=1000, solver="liblinear",
                                    class_weight="balanced", random_state=42),
        "rf":    RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                        class_weight="balanced", random_state=42),
        "gb":    GradientBoostingClassifier(random_state=42)
    }

    # pesos simples por desbalance
    pos_w = (len(y)/(2.0*(y==1).sum())) if (y==1).sum()>0 else 1.0
    neg_w = (len(y)/(2.0*(y==0).sum())) if (y==0).sum()>0 else 1.0
    sw = np.where(y==1, pos_w, neg_w)

    Xtr, Xte, ytr, yte, sw_tr, sw_te = train_test_split(
        X, y, sw, test_size=0.3, stratify=y, random_state=42
    )

    results = []
    for name, est in candidates.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        try:
            pipe.fit(Xtr, ytr, model__sample_weight=sw_tr)
        except TypeError:
            pipe.fit(Xtr, ytr)
        p1 = pipe.predict_proba(Xte)[:, 1]
        # Curvas y umbral óptimo por F1
        prec, rec, thr_pr = precision_recall_curve(yte, p1)
        f1s = np.where((prec+rec)>0, 2*prec*rec/(prec+rec), 0.0)
        idx = int(np.nanargmax(f1s[:-1])) if len(f1s) > 1 else 0
        thr_star = float(thr_pr[idx]) if len(thr_pr) else 0.5
        yhat = (p1 >= thr_star).astype(int)
        # Métricas
        auc = float(roc_auc_score(yte, p1))
        aupr = float(average_precision_score(yte, p1))
        acc = float(accuracy_score(yte, yhat))
        pre1 = float(precision_score(yte, yhat, zero_division=0))
        rec1 = float(recall_score(yte, yhat, zero_division=0))
        f1 = float(f1_score(yte, yhat, zero_division=0))
        cm = confusion_matrix(yte, yhat, labels=[0,1]).tolist()
        fpr, tpr, thr_roc = roc_curve(yte, p1)

        results.append({
            "name": name, "pipeline": pipe, "threshold": thr_star,
            "roc_auc": auc, "pr_auc": aupr, "accuracy": acc,
            "precision": pre1, "recall": rec1, "f1": f1,
            "confusion": cm,
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thr": thr_roc.tolist()},
            "pr_curve": {"precision": prec.tolist(), "recall": rec.tolist(),
                         "thr": (thr_pr.tolist() if hasattr(thr_pr, "tolist") else list(thr_pr))}
        })

    best = max(results, key=lambda r: r["roc_auc"])
    return best, results, num_cols, cat_cols, list(X.columns)

# ===== Artefactos =====
def save_artifacts(best, results, num_cols, cat_cols, all_cols):
    os.makedirs(OUT_DIR, exist_ok=True)

    # Modelo ganador
    _dump_model(best["pipeline"], os.path.join(OUT_DIR, "pipeline.pkl"))

    # Metadata para scoring en Streamlit
    meta = {
        "model_name": "hmeq_" + best["name"],
        "version": "v1",
        "target": TARGET,
        "threshold": float(best["threshold"]),
        "selection_metric": "roc_auc",
        "metrics_holdout": {
            "roc_auc": best["roc_auc"], "pr_auc": best["pr_auc"],
            "accuracy": best["accuracy"], "precision": best["precision"],
            "recall": best["recall"], "f1_at_best": best["f1"]
        },
        "inputs": [{"name": c, "type": ("number" if c in num_cols else "string"), "required": False}
                   for c in all_cols],
        "outputs": [{"name":"p_1","type":"number"},{"name":"label","type":"int"}]
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Tabla comparativa de modelos
    rows = []
    for r in results:
        rows.append({
            "model": r["name"], "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
            "accuracy": r["accuracy"], "precision": r["precision"],
            "recall": r["recall"], "f1": r["f1"], "threshold": r["threshold"]
        })
    # escribir CSV sin pandas
    def _csv(path, rows):
        if not rows: return
        keys = list(rows[0].keys())
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                f.write(",".join(str(row[k]) for k in keys) + "\n")
    _csv(os.path.join(OUT_DIR, "metrics_all.csv"), rows)

    # Métricas del mejor + confusion y curvas
    with open(os.path.join(OUT_DIR, "metrics_best.json"), "w", encoding="utf-8") as f:
        json.dump({k: best[k] for k in ["name","roc_auc","pr_auc","accuracy","precision","recall","f1","threshold"]},
                  f, indent=2)

    # Confusion matrix
    with open(os.path.join(OUT_DIR, "confusion_matrix_best.csv"), "w", encoding="utf-8") as f:
        f.write("pred_0,pred_1\n")
        f.write(",".join(str(x) for x in best["confusion"][0]) + "\n")
        f.write(",".join(str(x) for x in best["confusion"][1]) + "\n")

    # Curvas ROC y PR
    with open(os.path.join(OUT_DIR, "roc_curve_best.csv"), "w", encoding="utf-8") as f:
        f.write("fpr,tpr,threshold\n")
        for a,b,c in zip(best["roc_curve"]["fpr"], best["roc_curve"]["tpr"], best["roc_curve"]["thr"]):
            f.write(f"{a},{b},{c}\n")
    with open(os.path.join(OUT_DIR, "pr_curve_best.csv"), "w", encoding="utf-8") as f:
        f.write("precision,recall\n")
        for a,b in zip(best["pr_curve"]["precision"], best["pr_curve"]["recall"]):
            f.write(f"{a},{b}\n")

    # score.py (sin comillas triples; sin requests)
    score_lines = [
        "import os, threading, pandas as pd, json",
        "try:",
        "    import joblib as _joblib",
        "except Exception:",
        "    _joblib = None",
        "import pickle as _pickle",
        "_LOCK = threading.Lock()",
        "_MODEL = None",
        "_THRESHOLD = None",
        "def _read_threshold():",
        "    here = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()",
        "    try:",
        "        with open(os.path.join(here, 'metadata.json'), 'r', encoding='utf-8') as f:",
        "            return float(json.load(f).get('threshold', 0.5))",
        "    except Exception:",
        "        return 0.5",
        "def _fetch(url, timeout=20):",
        "    from urllib.request import urlopen, Request",
        "    req = Request(url, headers={'User-Agent':'Python'})",
        "    with urlopen(req, timeout=timeout) as r:",
        "        return r.read()",
        "def _ensure_pkl(path):",
        "    if os.path.exists(path):",
        "        return",
        "    url = os.environ.get('PIPELINE_URL')",
        "    if url:",
        "        data = _fetch(url)",
        "        open(path, 'wb').write(data)",
        "def _load():",
        "    global _MODEL, _THRESHOLD",
        "    if _MODEL is None:",
        "        with _LOCK:",
        "            if _MODEL is None:",
        "                here = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()",
        "                pkl = os.path.join(here, 'pipeline.pkl')",
        "                _ensure_pkl(pkl)",
        "                if _joblib is not None:",
        "                    _MODEL = _joblib.load(pkl)",
        "                else:",
        "                    _MODEL = _pickle.load(open(pkl, 'rb'))",
        "                th = os.environ.get('THRESHOLD')",
        "                _THRESHOLD = float(th) if th is not None else _read_threshold()",
        "    return _MODEL, _THRESHOLD",
        "def _score_one(d):",
        "    m, t = _load()",
        "    X = pd.DataFrame([d])",
        "    p1 = float(m.predict_proba(X)[0, 1])",
        "    return {'p_1': p1, 'label': int(p1 >= t)}",
        "def score(record):",
        "    if isinstance(record, list):",
        "        return [_score_one(r) for r in record]",
        "    return _score_one(record)",
    ]
    with open(os.path.join(OUT_DIR, "score.py"), "w", encoding="utf-8") as f:
        f.write("\n".join(score_lines))

    # requirements.txt mínimos
    with open(os.path.join(OUT_DIR, "requirements.txt"), "w") as f:
        f.write("pandas\nnumpy\nscikit-learn\njoblib\n")

# ===== Main =====
def main():
    print("RUN @", HERE, "| OUT_DIR =", OUT_DIR, "| DATA_MODE =", DATA_MODE)
    try:
        if DATA_MODE.lower() == "cas":
            df = load_from_cas()
        else:
            df = load_from_csv()
    except Exception as e:
        print("CAS/CSV falló:", e, "→ intento CSV de ejemplo")
        df = load_from_csv()

    if TARGET not in df.columns:
        raise RuntimeError("No encuentro '%s' en los datos" % TARGET)
    df[TARGET] = df[TARGET].astype(int)

    best, results, num_cols, cat_cols, all_cols = train_and_eval(df)
    save_artifacts(best, results, num_cols, cat_cols, all_cols)
    print("OK — Mejor modelo:", best["name"],
          " AUC=%.4f" % best["roc_auc"], " F1=%.4f" % best["f1"],
          " thr*=%.3f" % best["threshold"])
    print("Artefactos y métricas en:", OUT_DIR)

if __name__ == "__main__":
    main()
