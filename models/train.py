proc python;
submit;
import os, json, joblib, warnings
warnings.filterwarnings("ignore")

import pandas as pd, numpy as np, swat
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ---------- CONFIG ----------
CAS_LIB = os.getenv("CAS_LIB", "Public")
CAS_TABLE = os.getenv("CAS_TABLE", "HMEQ")
TARGET = os.getenv("TARGET", "BAD")           # positivo = 1
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.3"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# ---------- CAS ----------
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
if TARGET not in df.columns:
    raise ValueError(f"Target '{TARGET}' not found in {CAS_LIB}.{CAS_TABLE}")
df[TARGET] = df[TARGET].astype(int)

X = df.drop(columns=[TARGET]); y = df[TARGET].values
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# ---------- PREPRO ----------
from sklearn.preprocessing import OneHotEncoder
try:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.2
except TypeError:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn < 1.2

numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
if cat_cols:
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OHE)])
    preprocess = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])
else:
    preprocess = ColumnTransformer([("num", numeric, num_cols)])

# ---------- MODELOS ----------
candidates = {
    "logistic": LogisticRegression(max_iter=1000, solver="liblinear",
                                   class_weight="balanced", random_state=RANDOM_STATE),
    "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                 class_weight="balanced", random_state=RANDOM_STATE),
    "gb": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

# pesos por clase
pos_w = (len(y)/(2.0*(y==1).sum())) if (y==1).sum()>0 else 1.0
neg_w = (len(y)/(2.0*(y==0).sum())) if (y==0).sum()>0 else 1.0
sw = np.where(y==1, pos_w, neg_w)

Xtr, Xte, ytr, yte, sw_tr, sw_te = train_test_split(
    X, y, sw, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

def fit_eval(name, est):
    pipe = Pipeline([("preprocess", preprocess), ("model", est)])
    try:
        pipe.fit(Xtr, ytr, model__sample_weight=sw_tr)
    except TypeError:
        pipe.fit(Xtr, ytr)
    p1 = pipe.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, p1)
    aupr = average_precision_score(yte, p1)
    prec, rec, thr = precision_recall_curve(yte, p1)
    f1s = np.where((prec+rec)>0, 2*prec*rec/(prec+rec), 0.0)
    idx = int(np.nanargmax(f1s[:-1])) if len(f1s)>1 else 0
    best_thr = float(thr[idx]) if len(thr)>0 else 0.5
    f1 = f1_score(yte, (p1>=best_thr).astype(int), pos_label=1)
    return {"name":name,"auc":float(auc),"aupr":float(aupr),"f1_best":float(f1),"thr":best_thr,"pipeline":pipe}

results = [fit_eval(n,e) for n,e in candidates.items()]
best = max(results, key=lambda r: r["auc"])

# ---------- ARTEFACTOS ----------
joblib.dump(best["pipeline"], "pipeline.pkl")

def levels_for(col, cap=50):
    vals = pd.Series(col).dropna().astype(str).unique().tolist()
    return vals[:cap] if len(vals)>cap else vals

metadata = {
  "model_name": f"hmeq_{best['name']}",
  "version": "v1",
  "target": TARGET,
  "threshold": float(best["thr"]),              # umbral óptimo (evento=1)
  "selection_metric": "roc_auc",
  "metrics_holdout": {"roc_auc":best["auc"], "pr_auc":best["aupr"], "f1_at_best":best["f1_best"]},
  "inputs": [
    {"name": c, "type": ("number" if c in num_cols else "string"), "required": False,
     **({"levels": levels_for(X[c])} if c in cat_cols else {})}
    for c in X.columns
  ],
  "outputs": [{"name":"p_1","type":"number"},{"name":"label","type":"int"}]
}
with open("metadata.json","w",encoding="utf-8") as f:
    json.dump(metadata,f,indent=2,ensure_ascii=False)

# score.py con triple comilla: sin escapes ni concatenaciones
score_py = """
import os, threading, pandas as pd, joblib, json

_LOCK = threading.Lock()
_MODEL = None
_THRESHOLD = None

def _meta_thr():
    here = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    try:
        with open(os.path.join(here, "metadata.json"), "r", encoding="utf-8") as f:
            return float(json.load(f).get("threshold", 0.5))
    except Exception:
        return 0.5

def _ensure_pkl(p):
    if os.path.exists(p): return
    url = os.environ.get("PIPELINE_URL")
    if url:
        import requests
        r = requests.get(url, timeout=20); r.raise_for_status()
        with open(p, "wb") as f: f.write(r.content)

def _load():
    global _MODEL, _THRESHOLD
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                here = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
                pkl = os.path.join(here, "pipeline.pkl")
                _ensure_pkl(pkl)
                _MODEL = joblib.load(pkl)
                th = os.environ.get("THRESHOLD")
                _THRESHOLD = float(th) if th is not None else _meta_thr()
    return _MODEL, _THRESHOLD

def _score_one(d):
    m, t = _load()
    X = pd.DataFrame([d])
    p1 = float(m.predict_proba(X)[0, 1])
    return {"p_1": p1, "label": int(p1 >= t)}

def score(record):
    if isinstance(record, list):
        return [_score_one(r) for r in record]
    return _score_one(record)
"""

with open("score.py","w",encoding="utf-8") as f:
    f.write(score_py)

with open("requirements.txt","w") as f:
    f.write("pandas>=2.0\nnumpy>=1.24\nscikit-learn>=1.0\njoblib>=1.2\nrequests>=2.31\n")

print("OK — Artefactos: pipeline.pkl, metadata.json, score.py, requirements.txt")
endsubmit;
quit;
