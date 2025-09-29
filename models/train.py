# ==== Auto-ML ligero binario desde CAS (Public.HMEQ) con compat y trazas ====
import os, sys, json, textwrap, joblib, warnings, traceback
warnings.filterwarnings("ignore")

print("=== ENV INFO ===")
print("Python :", sys.version)
try:
    import sklearn, pandas as pd, numpy as np
    print("sklearn:", sklearn.__version__)
    print("pandas :", pd.__version__)
    print("numpy  :", np.__version__)
except Exception as _:
    pass
print("================")

try:
    import pandas as pd, numpy as np
    import swat

    # ---------- 0) Config ----------
    CAS_LIB      = os.getenv("CAS_LIB", "Public")
    CAS_TABLE    = os.getenv("CAS_TABLE", "HMEQ")
    TARGET       = os.getenv("TARGET", "BAD")       # evento positivo = 1
    BEST_METRIC  = os.getenv("BEST_METRIC", "roc_auc")
    TEST_SIZE    = float(os.getenv("TEST_SIZE", "0.3"))
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

    # ---------- 1) Leer de CAS ----------
    def connect_cas():
        try:
            s = swat.CAS(); s.serverstatus(); return s
        except Exception:
            host = os.environ.get("CAS_HOST") or os.environ.get("SAS_CAS_HOST")
            port = int(os.environ.get("CAS_PORT", "5570"))
            protocol = os.environ.get("CAS_PROTOCOL", "http")
            token = os.environ.get("SAS_SERVICES_TOKEN") or os.environ.get("ACCESS_TOKEN")
            if token: return swat.CAS(host, port, protocol=protocol, token=token)
            user = os.environ.get("CAS_USER"); pwd = os.environ.get("CAS_PASSWORD")
            return swat.CAS(host, port, protocol=protocol, username=user, password=pwd)

    s = connect_cas()
    df = s.CASTable(CAS_TABLE, caslib=CAS_LIB).to_frame()

    print(f"Tabla CAS leída: {CAS_LIB}.{CAS_TABLE} → shape={df.shape}")
    print("Columnas:", list(df.columns)[:20], "...")

    if TARGET not in df.columns:
        raise ValueError(f"No encuentro la columna {TARGET} en {CAS_LIB}.{CAS_TABLE}")

    df[TARGET] = df[TARGET].astype(int)  # positivo = 1
    X = df.drop(columns=[TARGET])
    y = df[TARGET].values

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    print(f"num_cols={len(num_cols)}  cat_cols={len(cat_cols)}")

    # ---------- 2) Preprocesamiento + modelos ----------
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        precision_recall_curve, log_loss
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    # OneHotEncoder compat: sparse_output (>=1.2) vs sparse (<1.2)
    try:
        from sklearn.preprocessing import OneHotEncoder
        _ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        OHE_KW = {"handle_unknown":"ignore", "sparse_output":False}
        print("OneHotEncoder: usando sparse_output=False")
    except TypeError:
        from sklearn.preprocessing import OneHotEncoder
        _ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        OHE_KW = {"handle_unknown":"ignore", "sparse":False}
        print("OneHotEncoder: fallback a sparse=False (sklearn < 1.2)")

    # HistGradientBoosting compat
    HGB = None
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        HGB = HistGradientBoostingClassifier
        print("HistGradientBoosting disponible.")
    except Exception:
        try:
            from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
            from sklearn.ensemble import HistGradientBoostingClassifier
            HGB = HistGradientBoostingClassifier
            print("HistGradientBoosting disponible (vía experimental).")
        except Exception:
            print("HistGradientBoosting NO disponible en esta versión.")

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**OHE_KW))
    ]) if cat_cols else None

    transformers = [("num", numeric, num_cols)] + (
        [("cat", categorical, cat_cols)] if cat_cols else []
    )
    preprocess = ColumnTransformer(transformers)

    # Modelos candidatos
    candidates = {
        "logistic": LogisticRegression(max_iter=1000, solver="liblinear",
                                       class_weight="balanced", random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                     class_weight="balanced", random_state=RANDOM_STATE),
        "gb": GradientBoostingClassifier(random_state=RANDOM_STATE)
    }
    if HGB is not None:
        candidates["hgb"] = HGB(random_state=RANDOM_STATE)

    from collections import defaultdict
    results = []

    # pesos por clase (para desbalance)
    pos_w = (len(y) / (2.0 * (y == 1).sum())) if (y == 1).sum() > 0 else 1.0
    neg_w = (len(y) / (2.0 * (y == 0).sum())) if (y == 0).sum() > 0 else 1.0
    sample_weight = np.where(y == 1, pos_w, neg_w)

    Xtr, Xte, ytr, yte, sw_tr, sw_te = train_test_split(
        X, y, sample_weight, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Split: train={Xtr.shape}  test={Xte.shape}")

    def fit_and_eval(name, est):
        pipe = Pipeline([("preprocess", preprocess), ("model", est)])
        # algunos estimadores aceptan sample_weight
        try:
            pipe.fit(Xtr, ytr, model__sample_weight=sw_tr)
        except TypeError:
            pipe.fit(Xtr, ytr)

        p1_te = pipe.predict_proba(Xte)[:, 1]
        roc = roc_auc_score(yte, p1_te)
        prc = average_precision_score(yte, p1_te)
        try:
            ll = log_loss(yte, np.c_[1 - p1_te, p1_te])
        except Exception:
            ll = float("nan")

        prec, rec, thr = precision_recall_curve(yte, p1_te)
        f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
        best_idx = int(np.nanargmax(f1s[:-1])) if len(f1s) > 1 else 0
        best_thr = float(thr[best_idx]) if len(thr) else 0.5
        f1 = f1_score(yte, (p1_te >= best_thr).astype(int), pos_label=1)

        results.append({
            "name": name, "roc_auc": float(roc), "pr_auc": float(prc),
            "logloss": float(ll), "f1_at_best": float(f1), "best_threshold": best_thr,
            "pipeline": pipe
        })
        print(f"[{name}] AUC={roc:.4f}  AUPRC={prc:.4f}  F1@best={f1:.4f}  thr*={best_thr:.3f}")

    for name, est in candidates.items():
        fit_and_eval(name, est)

    metric_key = {"roc_auc":"roc_auc","pr_auc":"pr_auc","f1":"f1_at_best"}.get(BEST_METRIC, "roc_auc")
    best = max(results, key=lambda r: r[metric_key])

    # ---------- 3) Guardar artefactos ----------
    joblib.dump(best["pipeline"], "pipeline.pkl")

    def levels_for(col, max_levels=50):
        vals = pd.Series(col).dropna().astype(str).unique().tolist()
        return vals[:max_levels] if len(vals) > max_levels else vals

    metadata = {
      "model_name": f"hmeq_{best['name']}",
      "version": "v1",
      "target": TARGET,
      "threshold": float(best["best_threshold"]),
      "selection_metric": metric_key,
      "metrics_holdout": {k: best[k] for k in ["roc_auc","pr_auc","logloss","f1_at_best"]},
      "inputs": [
        {
          "name": c,
          "type": ("number" if c in num_cols else "string"),
          "required": False,
          **({"levels": levels_for(X[c])} if c in cat_cols else {})
        } for c in X.columns
      ],
      "outputs": [{"name":"p_1","type":"number"},{"name":"label","type":"int"}]
    }
    with open("metadata.json","w",encoding="utf-8") as f:
        json.dump(metadata,f,indent=2,ensure_ascii=False)

    # informe de modelos
    all_models = [
        {k: (float(v) if isinstance(v, (int,float,np.floating)) else v)
         for k,v in r.items() if k != "pipeline"}
        for r in results
    ]
    with open("models_trained.json","w",encoding="utf-8") as f:
        json.dump({"best_by": metric_key, "models": all_models}, f, indent=2)

    # score.py
    score_py = textwrap.dedent("""
    import os, threading, pandas as pd, joblib, json
    _LOCK = threading.Lock()
    _MODEL = None
    _THRESHOLD = None
    def _load_meta_threshold():
        here = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
        try:
            with open(os.path.join(here,"metadata.json"),"r",encoding="utf-8") as f:
                return float(json.load(f).get("threshold",0.5))
        except Exception:
            return 0.5
    def _ensure_pipeline_local(path):
        if os.path.exists(path): return
        url = os.environ.get("PIPELINE_URL")
        if url:
            import requests
            r = requests.get(url, timeout=20); r.raise_for_status()
            with open(path,"wb") as f: f.write(r.content)
    def _load():
        global _MODEL, _THRESHOLD
        if _MODEL is None:
            with _LOCK:
                if _MODEL is None:
                    here = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
                    pkl = os.path.join(here,"pipeline.pkl")
                    _ensure_pipeline_local(pkl)
                    _MODEL = joblib.load(pkl)
                    _THRESHOLD = float(os.environ.get("THRESHOLD","nan"))
                    if not (_THRESHOLD == _THRESHOLD):  # NaN
                        _THRESHOLD = _load_meta_threshold()
        return _MODEL, _THRESHOLD
    def _score_one(d):
        model, thr = _load()
        X = pd.DataFrame([d])
        p1 = float(model.predict_proba(X)[0,1])
        return {"p_1": p1, "label": int(p1 >= thr)}
    def score(record):
        if isinstance(record, list):
            return [_score_one(r) for r in record]
        return _score_one(record)
    """).strip()
    with open("score.py","w",encoding="utf-8") as f:
        f.write(score_py)

    with open("requirements.txt","w") as f:
        f.write("\n".join([
            "pandas>=2.0",
            "numpy>=1.24",
            "scikit-learn>=1.0",  # compat amplia
            "joblib>=1.2",
            "requests>=2.31"
        ]) + "\n")

    print("Listo ✅")
    print(f"Mejor: {metadata['model_name']} | métrica={metric_key} | umbral={metadata['threshold']:.4f}")

except Exception as e:
    print("❌ ERROR EN EL SCRIPT PYTHON")
    traceback.print_exc()
    raise
