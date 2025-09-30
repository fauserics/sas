# app.py
# Real Time Scoring App (powered by SAS)
# Engines:
#  1) Local: translate SAS DATA step score code -> Python and run
#  2) Viya REST: call MAS scoring endpoint

import os
import json
import pandas as pd
import streamlit as st

from sas_code_translator import compile_sas_score, score_dataframe as score_df_local
from viya_mas_client import score_row_via_rest  # needs VIYA_URL, MAS_MODULE_ID, token/creds

APP_TITLE = "Real Time Scoring App (powered by SAS)"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

SCORE_DIR = "score"
DATA_DIR = "data"

# ---------- helpers ----------
def find_sas_score_file() -> str | None:
    preferred = ["score_code.sas", "model_score.sas", "model_score_code.sas", "dmcas_scorecode.sas"]
    for name in preferred:
        p = os.path.join(SCORE_DIR, name)
        if os.path.isfile(p):
            return p
    if not os.path.isdir(SCORE_DIR):
        return None
    for fn in sorted(os.listdir(SCORE_DIR)):
        if fn.lower().endswith(".sas"):
            return os.path.join(SCORE_DIR, fn)
    return None

def load_file_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@st.cache_data
def load_input_vars() -> list[str]:
    p = os.path.join(SCORE_DIR, "inputVar.json")
    if not os.path.isfile(p):
        return []
    try:
        data = json.load(open(p, "r", encoding="utf-8"))
        if isinstance(data, list) and data and isinstance(data[0], dict) and "name" in data[0]:
            return [d["name"] for d in data]
        if isinstance(data, dict):
            if "variables" in data and isinstance(data["variables"], list):
                return [d.get("name") for d in data["variables"] if "name" in d]
            if "inputVariables" in data and isinstance(data["inputVariables"], list):
                return [d.get("name") for d in data["inputVariables"] if "name" in d]
    except Exception:
        pass
    return []

@st.cache_data
def load_sample_df(expected_cols: list[str]) -> pd.DataFrame:
    for cand in [os.path.join(DATA_DIR, "sample.csv"), os.path.join(DATA_DIR, "hmeq.csv")]:
        if os.path.isfile(cand):
            try:
                df = pd.read_csv(cand)
                return df.drop(columns=["BAD"], errors="ignore")
            except Exception:
                pass
    return pd.DataFrame(columns=expected_cols)

def parse_mas_outputs(resp_json: dict, threshold: float = 0.5) -> tuple[float, int]:
    prob = None; label = None
    outputs = resp_json.get("outputs")
    if isinstance(outputs, list):
        d = {o.get("name"): o.get("value") for o in outputs if isinstance(o, dict)}
        prob = d.get("EM_EVENTPROBABILITY") or d.get("P_BAD1") or d.get("P_1") or d.get("EM_PREDICTION")
        label = d.get("EM_CLASSIFICATION") or d.get("I_BAD") or d.get("LABEL")
    if prob is None:
        for k in ("EM_EVENTPROBABILITY","P_BAD1","P_1","EM_PREDICTION","probability","score"):
            if k in resp_json:
                prob = resp_json[k]; break
    if label is None:
        for k in ("EM_CLASSIFICATION","I_BAD","LABEL"):
            if k in resp_json:
                label = resp_json[k]; break
    if prob is None:
        prob = 0.0
    try:
        label = int(label) if label is not None and str(label).isdigit() else (1 if float(prob) >= threshold else 0)
    except Exception:
        label = 1 if float(prob) >= threshold else 0
    return float(prob), int(label)

def build_single_record_form(fields: list[str], sample: pd.DataFrame) -> dict:
    defaults = {}
    if not sample.empty:
        for c in fields:
            if c in sample.columns:
                s = sample[c]
                if pd.api.types.is_numeric_dtype(s):
                    defaults[c] = float(s.dropna().median()) if s.notna().any() else 0.0
                else:
                    defaults[c] = str(s.dropna().mode().iloc[0]) if s.notna().any() else ""
            else:
                defaults[c] = 0.0
    else:
        for c in fields:
            defaults[c] = 0.0

    left, right = st.columns(2)
    row = {}
    for i, c in enumerate(fields):
        tgt = left if i % 2 == 0 else right
        if c in sample.columns and pd.api.types.is_numeric_dtype(sample[c]):
            row[c] = tgt.number_input(c, value=float(defaults.get(c, 0.0)))
        else:
            opts = []
            if c in sample.columns and not sample.empty:
                vals = sample[c].dropna().astype(str).unique().tolist()
                if 1 <= len(vals) <= 50:
                    opts = sorted(vals)
            row[c] = (tgt.selectbox(c, opts, index=0) if opts
                      else tgt.text_input(c, value=str(defaults.get(c, ""))))
    return row

def is_ds2_astore(code: str) -> bool:
    sigs = ("package score", "dcl package score", "scoreRecord()", "method run()", "enddata;")
    up = code
    return any(s in up for s in sigs)

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### Settings")
    thr = st.slider("Decision threshold (BAD=1)", 0.0, 1.0, 0.50, 0.01)
    st.markdown("---")
    st.markdown("**Viya REST env**")
    st.caption(f"VIYA_URL: {os.getenv('VIYA_URL') or '(not set)'}")
    st.caption(f"MAS_MODULE_ID: {os.getenv('MAS_MODULE_ID') or '(not set)'}")
    has_token = bool(os.getenv("BEARER_TOKEN") or os.getenv("SAS_SERVICES_TOKEN") or os.getenv("VIYA_USER"))
    st.caption(f"Auth: {'configured' if has_token else 'missing'}")

# ---------- load inputs and sample ----------
expected_from_json = load_input_vars()
sas_path = find_sas_score_file()
if not sas_path:
    st.warning("No SAS score file found under ./score. Place your exported DATA step (*.sas) there.")
else:
    st.caption(f"Using SAS score file: `{sas_path}`")

sample_df = load_sample_df(expected_from_json)

# ---------- session state ----------
if "score_fn" not in st.session_state:
    st.session_state.score_fn = None
if "score_py" not in st.session_state:
    st.session_state.score_py = ""
if "expected_cols" not in st.session_state:
    st.session_state.expected_cols = expected_from_json

# ---------- translation ----------
st.markdown("## 1) Translate SAS score code to Python")
col_t1, col_t2 = st.columns([1, 3])
with col_t1:
    can_translate = st.button("Translate SAS → Python", type="primary", use_container_width=True, disabled=not bool(sas_path))
with col_t2:
    st.caption("Translates your SAS DATA step score code into a Python function (no training required).")

sas_is_ds2 = False
if can_translate and sas_path:
    code = load_file_text(sas_path)
    sas_is_ds2 = is_ds2_astore(code)
    if sas_is_ds2:
        st.warning("This SAS file looks like DS2/ASTORE (e.g., scoreRecord). Local translation is not supported. Use 'Score on Viya (REST)'.")
    else:
        try:
            score_fn, py_code, expected = compile_sas_score(
                code, func_name="sas_score", expected_inputs=(expected_from_json or None)
            )
            st.session_state.score_fn = score_fn
            st.session_state.score_py = py_code
            st.session_state.expected_cols = expected_from_json or expected or []
            st.success(f"Translation OK. Found {len(st.session_state.expected_cols)} input fields.")
        except Exception as e:
            st.error(f"Translation failed: {e}")

with st.expander("Show generated Python score code", expanded=False):
    if st.session_state.score_py:
        st.code(st.session_state.score_py, language="python")
        save_py = st.toggle("Save generated Python to file (score/translated_score.py)", value=False)
        if save_py:
            os.makedirs("score", exist_ok=True)
            outp = os.path.join("score", "translated_score.py")
            with open(outp, "w", encoding="utf-8") as f:
                f.write(st.session_state.score_py)
            st.success(f"Saved: {outp}")
    else:
        st.info("Translate first to view the generated code.")

# ---------- tabs ----------
tabs = st.tabs(["Single case", "CSV batch", "Info"])

# ---- Single case ----
with tabs[0]:
    st.markdown("## 2) Provide inputs")
    fields = st.session_state.expected_cols or list(sample_df.columns)
    if not fields:
        st.info("No input schema found. Add score/inputVar.json or a data/sample.csv.")
    row = build_single_record_form(fields, sample_df)

    st.markdown("## 3) Score")
    cA, cB = st.columns(2)
    do_local = cA.button("Score locally (Python)", type="primary", use_container_width=True,
                         disabled=(st.session_state.score_fn is None))
    do_rest  = cB.button("Score on Viya (REST)", use_container_width=True)

    results = {}
    if do_local:
        try:
            df_in = pd.DataFrame([row]).reindex(columns=fields, fill_value=None)
            scored = score_df_local(st.session_state.score_fn, df_in, threshold=thr)
            p = float(scored["prob_BAD"].iloc[0]); lbl = int(scored["label_BAD"].iloc[0])
            results["Local (Python)"] = (p, lbl)
            st.success("Local scoring done.")
            c1, c2 = st.columns(2)
            c1.metric("Probability of BAD (local)", f"{p:0.4f}")
            c2.metric("Predicted label (local)", "1" if lbl == 1 else "0")
            st.dataframe(scored)
        except Exception as e:
            st.error(f"Local scoring failed: {e}")

    if do_rest:
        try:
            with st.spinner("Calling Viya..."):
                resp = score_row_via_rest(row)
            p, lbl = parse_mas_outputs(resp, threshold=thr)
            results["Viya (REST)"] = (p, lbl)
            st.success("Viya scoring done.")
            c1, c2 = st.columns(2)
            c1.metric("Probability of BAD (Viya)", f"{p:0.4f}")
            c2.metric("Predicted label (Viya)", "1" if lbl == 1 else "0")
            st.json(resp)
        except Exception as e:
            st.error(f"Viya REST error: {e}")

    if len(results) == 2:
        (pl, ll) = results["Local (Python)"]
        (pr, lr) = results["Viya (REST)"]
        diff = abs(pl - pr); mismatch = int(ll != lr)
        st.markdown("### Comparison")
        d1, d2 = st.columns(2)
        d1.metric("Δ Probability |local - viya|", f"{diff:0.6f}")
        d2.metric("Label mismatch (0/1)", f"{mismatch}")

# ---- CSV batch ----
with tabs[1]:
    st.markdown("## Batch scoring")
    up = st.file_uploader("Upload a CSV with the input schema", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            fields = st.session_state.expected_cols or list(df.columns)
            for c in fields:
                if c not in df.columns:
                    df[c] = None
            df = df[fields]

            c1, c2 = st.columns(2)
            do_local_csv = c1.button("Score CSV locally (Python)",
                                     use_container_width=True, disabled=(st.session_state.score_fn is None))
            do_rest_csv  = c2.button("Score CSV on Viya (REST)", use_container_width=True)

            if do_local_csv:
                try:
                    scored = score_df_local(st.session_state.score_fn, df, threshold=thr)
                    st.success(f"Scored locally: {len(scored)} rows.")
                    st.dataframe(scored.head(50))
                    st.download_button(
                        "Download local-scored CSV",
                        data=scored.to_csv(index=False).encode("utf-8"),
                        file_name="scored_local.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Local batch failed: {e}")

            if do_rest_csv:
                rows = []
                prog = st.progress(0)
                n = len(df)
                for i, (_, r) in enumerate(df.iterrows(), start=1):
                    try:
                        resp = score_row_via_rest(r.to_dict())
                        p, lbl = parse_mas_outputs(resp, threshold=thr)
                    except Exception:
                        p, lbl = float("nan"), None
                    rec = r.to_dict()
                    rec["prob_BAD_rest"] = p
                    rec["label_BAD_rest"] = lbl
                    rows.append(rec)
                    if i % 5 == 0 or i == n:
                        prog.progress(int(i*100/n))
                scored_rest = pd.DataFrame(rows)
                st.success(f"Scored on Viya: {len(scored_rest)} rows.")
                st.dataframe(scored_rest.head(50))
                st.download_button(
                    "Download Viya-scored CSV",
                    data=scored_rest.to_csv(index=False).encode("utf-8"),
                    file_name="scored_viya.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"CSV error: {e}")

# ---- Info ----
with tabs[2]:
    st.markdown("## Info")
    st.write("- SAS score file:", f"`{sas_path}`" if sas_path else "_not found_")
    st.write("- Inputs from inputVar.json:", len(expected_from_json))
    st.write("- Current expected inputs:", len(st.session_state.expected_cols))
    if st.session_state.expected_cols:
        with st.expander("Show expected input fields"):
            st.code("\n".join(st.session_state.expected_cols))
    st.write("- Sample data rows:", len(sample_df))
    if not sample_df.empty:
        with st.expander("Preview sample data"):
            st.dataframe(sample_df.head(20))
    st.caption("Tip: for DS2/ASTORE (.sas with package/scoreRecord), use Viya REST mode.")
