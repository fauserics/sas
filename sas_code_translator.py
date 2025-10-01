# app.py
# Real Time Scoring App (powered by SAS)
# Engines:
#  1) GitHub (Python): translate SAS DATA step score code -> Python and run locally
#  2) Viya REST: call MAS scoring endpoint

import os
import re
import json
import pandas as pd
import streamlit as st

from sas_code_translator import compile_sas_score, score_dataframe as score_df_local
from viya_mas_client import score_row_via_rest  # needs VIYA_URL, MAS_MODULE_ID, token/creds

APP_TITLE = "Real Time Scoring App (powered by SAS)"
ENGINE_LABEL = "GitHub (Python)"

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

def _sanitize_var(v: str) -> str:
    v = v.strip()
    m = re.match(r"^'([^']+)'\s*n$", v, flags=re.I)
    if m:
        v = m.group(1)
    v = re.sub(r"\W", "_", v)
    return v

def extract_expected_inputs_from_sas(code: str) -> list[str]:
    miss_names = set(_sanitize_var(x) for x in re.findall(r"missing\(\s*([A-Za-z_][\w']*)\s*\)", code, flags=re.I))
    if re.search(r"\bREASON\b", code, flags=re.I):
        miss_names.add("REASON")
    drop_prefixes = ("_", "EM_", "P_", "I_", "va__d__E_")
    keep = [n for n in miss_names if n and not n.upper().startswith(drop_prefixes)]
    return sorted(keep, key=str.upper)

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

def parse_mas_outputs(resp_json: dict, threshold: float = 0.5) -> tuple[float, int | None]:
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
    try:
        p = None if prob is None else float(prob)
    except Exception:
        p = None
    try:
        lbl = None if label is None else (int(label) if str(label).isdigit() else (1 if (p is not None and p >= threshold) else 0))
    except Exception:
        lbl = (1 if (p is not None and p >= threshold) else 0)
    return (p if p is not None else float("nan")), lbl

def build_single_record_form(fields: list[str], sample: pd.DataFrame, key_prefix: str) -> dict:
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
        key = f"{key_prefix}__{c}"
        if c in sample.columns and pd.api.types.is_numeric_dtype(sample[c]):
            row[c] = tgt.number_input(c, value=float(defaults.get(c, 0.0)), key=key)
        else:
            opts = []
            if c in sample.columns and not sample.empty:
                vals = sample[c].dropna().astype(str).unique().tolist()
                if 1 <= len(vals) <= 50:
                    opts = sorted(vals)
            if opts:
                row[c] = tgt.selectbox(c, opts, index=0, key=key)
            else:
                row[c] = tgt.text_input(c, value=str(defaults.get(c, "")), key=key)
    return row

def is_ds2_astore(code: str) -> bool:
    sigs = ("package score", "dcl package score", "scoreRecord()", "method run()", "enddata;")
    return any(s in code for s in sigs)

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
    # Opcional: recargar inputVar.json sin reiniciar
    if st.button("Reload inputVar.json"):
        load_input_vars.clear()
        st.session_state.expected_cols = load_input_vars()
        st.success("inputVar.json reloaded.")

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
    st.caption(f"Translates your SAS DATA step score code into a Python function (runs in {ENGINE_LABEL}).")

sas_is_ds2 = False
if can_translate and sas_path:
    raw_code = load_file_text(sas_path)
    sas_is_ds2 = is_ds2_astore(raw_code)
    if sas_is_ds2:
        st.warning("This SAS file looks like DS2/ASTORE (e.g., scoreRecord). Translation is not supported. Use 'Score on Viya (REST)'.")
    else:
        try:
            expected_auto = extract_expected_inputs_from_sas(raw_code)
            score_fn, py_code, expected = compile_sas_score(
                raw_code, func_name="sas_score",
                expected_inputs=(expected_from_json or expected_auto or None)
            )
            st.session_state.score_fn = score_fn
            st.session_state.score_py = py_code
            st.session_state.expected_cols = expected_from_json or expected_auto or expected or []
            st.success(f"Translation OK. Found {len(st.session_state.expected_cols)} input fields.")
        except Exception as e:
            st.error(f"Translation failed: {e}")

with st.expander("Show generated Python score code", expanded=False):
    if st.session_state.score_py:
        st.code(st.session_state.score_py, language="python")
        save_py = st.toggle("Save generated Python to file (score/translated_score.py)", value=False, key="save_py_toggle")
        if save_py:
            os.makedirs("score", exist_ok=True)
            outp = os.path.join("score", "translated_score.py")
            with open(outp, "w", encoding="utf-8") as f:
                f.write(st.session_state.score_py)
            st.success(f"Saved: {outp}")
    else:
        st.info("Translate first to view the generated code.")

# ---------- tabs ----------
tabs = st.tabs([f"Single case — {ENGINE_LABEL}", "Single case — Viya (REST)", "CSV batch", "Info"])

# ---- Single case — GitHub (Python) ----
with tabs[0]:
    st.markdown(f"## 2) Provide inputs for {ENGINE_LABEL}")
    fields = st.session_state.expected_cols or list(sample_df.columns)
    if not fields:
        st.info("No input schema found. Add score/inputVar.json or a data/sample.csv.")
    row = build_single_record_form(fields, sample_df, key_prefix="gh_single")

    st.markdown(f"## 3) Score in {ENGINE_LABEL}")
    do_local = st.button(f"Score in {ENGINE_LABEL}", type="primary", use_container_width=True, key="btn_gh_single",
                         disabled=(st.session_state.score_fn is None))

    if do_local:
        try:
            df_in = pd.DataFrame([row]).reindex(columns=fields, fill_value=None)
            scored = score_df_local(st.session_state.score_fn, df_in, threshold=thr)
            # ---- SAFE CASTS (evita int(None)) ----
            p_raw = scored["prob_BAD"].iloc[0] if "prob_BAD" in scored else float("nan")
            lbl_raw = scored["label_BAD"].iloc[0] if "label_BAD" in scored else None
            p = None if (pd.isna(p_raw)) else float(p_raw)
            lbl = None
            if lbl_raw is not None and not (isinstance(lbl_raw, float) and pd.isna(lbl_raw)):
                try:
                    lbl = int(lbl_raw)
                except Exception:
                    lbl = None

            if p is None:
                st.warning("Probability is NaN (likely missing required inputs for the SAS score). Check the form/inputVar.json.")
            else:
                st.success(f"Scored in {ENGINE_LABEL}.")
            c1, c2 = st.columns(2)
            c1.metric(f"Probability of BAD ({ENGINE_LABEL})", f"{(p if p is not None else float('nan')):0.4f}" if p is not None else "—")
            c2.metric(f"Predicted label ({ENGINE_LABEL})", "1" if lbl == 1 else ("0" if lbl == 0 else "—"))
            st.dataframe(scored)
        except Exception as e:
            st.error(f"{ENGINE_LABEL} scoring failed: {e}")

# ---- Single case — Viya (REST) ----
with tabs[1]:
    st.markdown("## 2) Provide inputs for Viya (REST)")
    fields = st.session_state.expected_cols or list(sample_df.columns)
    if not fields:
        st.info("No input schema found. Add score/inputVar.json or a data/sample.csv.")
    row_rest = build_single_record_form(fields, sample_df, key_prefix="viya_single")

    st.markdown("## 3) Score on Viya (REST)")
    do_rest  = st.button("Score on Viya (REST)", use_container_width=True, key="btn_viya_single")

    if do_rest:
        try:
            with st.spinner("Calling Viya..."):
                resp = score_row_via_rest(row_rest)
            p, lbl = parse_mas_outputs(resp, threshold=thr)
            st.success("Viya scoring done.")
            c1, c2 = st.columns(2)
            c1.metric("Probability of BAD (Viya)", f"{p:0.4f}" if not pd.isna(p) else "—")
            c2.metric("Predicted label (Viya)", "1" if lbl == 1 else ("0" if lbl == 0 else "—"))
            st.json(resp)
        except Exception as e:
            st.error(f"Viya REST error: {e}")

# ---- CSV batch ----
with tabs[2]:
    st.markdown("## Batch scoring")
    up = st.file_uploader("Upload a CSV with the input schema", type=["csv"], key="uploader_csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            fields = st.session_state.expected_cols or list(df.columns)
            for c in fields:
                if c not in df.columns:
                    df[c] = None
            df = df[fields]

            c1, c2 = st.columns(2)
            do_local_csv = c1.button(f"Score CSV in {ENGINE_LABEL}",
                                     use_container_width=True, disabled=(st.session_state.score_fn is None), key="btn_gh_csv")
            do_rest_csv  = c2.button("Score CSV on Viya (REST)", use_container_width=True, key="btn_viya_csv")

            if do_local_csv:
                try:
                    scored = score_df_local(st.session_state.score_fn, df, threshold=thr)
                    st.success(f"Scored in {ENGINE_LABEL}: {len(scored)} rows.")
                    st.dataframe(scored.head(50))
                    st.download_button(
                        f"Download {ENGINE_LABEL} CSV",
                        data=scored.to_csv(index=False).encode("utf-8"),
                        file_name="scored_github_python.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_gh_csv"
                    )
                except Exception as e:
                    st.error(f"{ENGINE_LABEL} batch failed: {e}")

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
                    use_container_width=True,
                    key="download_viya_csv"
                )
        except Exception as e:
            st.error(f"CSV error: {e}")

# ---- Info ----
with tabs[3]:
    st.markdown("## Info")
    sas_path = find_sas_score_file()
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
    st.caption(f"Tip: if the SAS file is DS2/ASTORE (contains scoreRecord/package), use the Viya REST tab. Otherwise, translate and score in {ENGINE_LABEL}.")
