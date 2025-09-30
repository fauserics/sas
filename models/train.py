# models/train.py — simple y compatible (SAS Studio / GitHub / local)
import os, json, warnings, joblib
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

# ===== Carga de datos =====
def load_from_cas():
    try:
        import swat
    except Exception as e:
        raise RuntimeError("SWAT no disponible; usá DATA_MODE=csv o instalá swat") from e
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
    import pandas as pd, io, requests
    if CSV_PATH and os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    url = CSV_URL or "https://raw.githubusercontent.com/selva86/datasets/master/HMEQ.csv"
    r = requests.get(url, timeout=30); r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

# ===== Modelo =====
def _make_ohe():
    from sklearn.preprocessing import OneHotEncoder
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn <1.2

def train(df):
    import numpy as np, pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    y = df[TARGET].astype(int).values
    X = df.drop(columns=[TARGET])

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

    # si por algún motivo no detectó columnas, crea un paso “identidad”
    from sklearn.compose import ColumnTransformer
    if not transformers:
        transformers = [("num", numeric, [])]

    pre = ColumnTransformer(transformers)

    clf = Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(max_iter=1000, solver="liblinear",
                                     class_weight="balanced", random_state=42))
    ])
    clf.fit(X, y)
    return clf, num_cols, cat_cols, list(X.columns)

# ===== Artefactos =====
def save_artifacts(model, num_cols, cat_cols, all_cols):
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) pipeline.pkl
    joblib.dump(model, os.path.join(OUT_DIR, "pipeline.pkl"))

    # 2) metadata.json
    meta = {
        "model_name": "hmeq_logit",
        "version": "v1",
        "target": TARGET,
        "threshold": 0.5,
        "inputs": [
            {"name": c, "type": ("number" if c in num_cols else "string"), "required": False}
            for c in all_cols
        ],
        "outputs": [{"name":"p_1","type":"number"},{"name":"label","type":"int"}]
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 3) score.py (sin comillas triples; lista de líneas)
    score_lines = [
        "import os, threading, pandas as pd, joblib, json",
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
        "def _ensure_pkl(path):",
        "    if os.path.exists(path):",
        "        return",
        "    url = os.environ.get('PIPELINE_URL')",
        "    if url:",
        "        import requests",
        "        r = requests.get(url, timeout=20); r.raise_for_status()",
        "        open(path, 'wb').write(r.content)",
        "def _load():",
        "    global _MODEL, _THRESHOLD",
        "    if _MODEL is None:",
        "        with _LOCK:",
        "            if _MODEL is None:",
        "                here = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()",
        "                pkl = os.path.join(here, 'pipeline.pkl')",
        "                _ensure_pkl(pkl)",
        "                _MODEL = joblib.load(pkl)",
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

    # 4) requirements.txt
    with open(os.path.join(OUT_DIR, "requirements.txt"), "w") as f:
        f.write("pandas>=2.0\nnumpy>=1.24\nscikit-learn>=1.0\njoblib>=1.2\nrequests>=2.31\n")

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

    model, num_cols, cat_cols, all_cols = train(df)
    save_artifacts(model, num_cols, cat_cols, all_cols)
    print("Listo ✓ Artefactos en:", OUT_DIR)

if __name__ == "__main__":
    main()
