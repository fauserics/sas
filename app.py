# app.py
# Real Time Scoring App (powered by SAS)

import os
import re
import json
import math
import pandas as pd
import streamlit as st

# Fuerzo el fallback del traductor (scorer logístico tolerante y sin NaN)
os.environ["SC_FORCE_FALLBACK"] = "1"

# IMPORTS (forma segura, sin alias largo en la misma línea)
import sas_code_translator as sct
try:
    from viya_mas_client import score_row_via_rest  # requiere VIYA_URL, MAS_MODULE_ID y auth
except Exception:
    # Si no existe, defino un stub para que la app cargue igual
    def score_row_via_rest(_row):
        raise RuntimeError("viya_mas_client not available. Set VIYA_URL/MAS_MODULE_ID or add the module.")

APP_TITLE = "Real Time Scoring App (powered by SAS)"
ENGINE_LABEL = "GitHub (Python)"

# Forzar estas variables como categóricas aun si el .sas no trae niveles (case-insensitive)
FORCE_CAT = {"reason"}
DEFAULT_LEVELS = {"reason": ["DebtCon", "HomeImp"]}

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

SCORE_DIR = "score"
DATA_DIR = "data"

# ---------- helpers ----------
def find_sas_score_file():
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

def load_file_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@st.cache_data
def load_input_vars():
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

def _sanitize_var(v):
    v = v.strip()
    m = re.match(r"^'([^']+)'\s*n$", v, flags=re.I)
    if m:
        v = m.group(1)
    v = re.sub(r"\W", "_", v)
    return v

def extract_expected_inputs_from_sas(code):
    miss_names = set(_sanitize_var(x) for x in re.findall(r"missing\(\s*([A-Za-z_][\w']*)\s*\)", code, flags=re.I))
    # si el código usa REASON de alguna forma, incluíla
    if re.search(r"\bREASON\b", code, flags=re.I) or re.search(r"\b_REASO[N_]\b", code, flags=re.I):
        miss_names.add("REASON")
    drop_prefixes = ("_", "EM_", "P_", "I_", "va__d__E_")
    keep = [n for n in miss_names if n and not n.upper().startswith(drop_prefixes)]
    return sorted(keep, key=str.upper)

def parse_categorical_levels_from_sas(code):
    """Extrae niveles de bloques select(VAR); when('...') ... end; y mapea alias con y sin '_'."""
    cat = {}
    for m in re.finditer(r"select\s*\(\s*([A-Za-z_]\w*)\s*\)\s*;(.+?)end\s*;", code, flags=re.I|re.S):
        var = m.group(1).strip()
        block = m.group(2)
        levels = re.findall(r"when\s*\(\s*'([^']+)'\s*\)", block, flags=re.I)
        if levels:
            lv = sorted(list(dict.fromkeys(levels)))
            cat[var] = lv
            base = var.lstrip("_")
            if base != var:
                cat[base] = lv
    return cat

def build_cat_levels_ci(cat_levels):
    """Mapa case-insensitive para categóricas, incluyendo alias con y sin '_'."""
    ci = {}
    for k, v in cat_levels.items():
        ci[k.lower()] = v
        base = k.lstrip("_").lower()
        ci[base] = v
        ci["_"+base] = v
    # Forzar categóricas aunque el .sas no las indique (e.g. REASON)
    for f in FORCE_CAT:
        if f not in ci and "_"+f not in ci:
            if f in DEFAULT_LEVELS:
                ci[f] = DEFAULT_LEVELS[f]
                ci["_"+f] = DEFAULT_LEVELS[f]
    return ci

@st.cache_data
def load_sample_df(expected_cols):
    for cand in [os.path.join(DATA_DIR, "sample.csv"), os.path.join(DATA_DIR, "hmeq.csv")]:
        if os.path.isfile(cand):
            try:
                df = pd.read_csv(cand)
                return df.drop(columns=["BAD"], errors="ignore")
            except Exception:
                pass
    return pd.DataFrame(columns=expected_cols)

def parse_mas_outputs(resp_json, threshold=0.5):
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

def build_single_record_form(fields, sample, cat_levels_ci, key_prefix):
    """Form: selectbox para categóricas (case-insensitive) y number_input para numéricas.
       Si se fuerza categórica sin niveles, usa text_input."""
    row = {}
    left, right = st.columns(2)
    for i, c in enumerate(fields):
        tgt = left if i % 2 == 0 else right
        key = f"{key_prefix}__{c}"
        cname = c.lower()
        levels = cat_levels_ci.get(cname) or cat_levels_ci.get("_"+cname)
        forced = cname in FORCE_CAT or ("_"+cname) in FORCE_CAT
        if levels:
            row[c] = tgt.selectbox(c, levels, index=0, key=key)
        elif forced:
            sugg = []
            if c in sample.columns and sample[c].dtype == object:
                sugg = sorted(list(pd.Series(sample[c]).dropna().unique()))[:20]
            if not sugg:
                sugg = DEFAULT_LEVELS.get(cname, [])
            if sugg:
                row[c] = tgt.selectbox(c, sugg, index=0, key=key)
            else:
                row[c] = tgt.text_input(c, value="", key=key)
        else:
            default = 0.0
            if c in sample.columns and pd.api.types.is_numeric_dtype(sample[c]):
                s = sample[c]
                default = float(s.dropna().median()) if s.notna().any() else 0.0
            row[c] = tgt.number_input(c, value=default, key=key)
    return row

def is_ds2_astore(code):
    sigs = ("package score", "dcl package score", "scoreRecord()", "method run()", "enddata;")
    return any(s in code for s in sigs)

def list_missing(fields, row):
    missing_now = []
    for c in fields:
        v = row.get(c, None)
        if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v.strip() == ""):
            missing_now.append(c)
    return missing_now

def copy_aliases_inplace_ci(row, cat_levels_ci):
    """Copia REASON <-> _REASON_ de forma case-insensitive si falta uno de los dos."""
    names = set()
    for k in list(cat_levels_ci.keys()):
        base = k.lstrip("_")
        names.add(base.lower()); names.add("_"+base.lower())
    for f in FORCE_CAT:
        names.add(f); names.add("_"+f)
    # helpers CI
    def get_ci(d, key):
        for kk in d.keys():
            if kk.lower() == key.lower(): return d[kk], kk
        return None, None
    # copiar
    for nm in names:
        if not nm.startswith("_"):
            base = nm
            alias = "_"+nm
            v_base, _ = get_ci(row, base)
            v_alias, _ = get_ci(row, alias)
            if v_base is not None and (v_alias is None or v_alias == ""):
                row[alias] = v_base
            elif v_alias is not None and (v_base is None or v_base == ""):
                row[base] = v_alias

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### Settings")
    thr = st.slider("Decision threshold (BAD=1)", 0.0, 1.0, 0.50, 0.01)
    debug = st.toggle("Debug mode", value=False)
    st.markdown("---")
    st.markdown("**Viya REST env**")
    st.caption(f"VIYA_URL: {os.getenv('VIYA_URL') or '(not set)'}")
    st.caption(f"MAS_MODULE_ID: {os.getenv('MAS_MODULE_ID') or '(not set)'}")
    has_token = bool(os.getenv("BEARER_TOKEN") or os.getenv("SAS_SERVICES_TOKEN") or os.getenv("VIYA_USER"))
    st.caption(f"Auth: {'configured' if has_token else 'missing'}")
    if st.button("Reload inputVar.json", key="reload_json_btn"):
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
if "cat_levels" not in st.session_state:
    st.session_state.cat_levels = {}
if "cat_levels_ci" not in st.session_state:
    st.session_state.cat_levels_ci = {}

# ---------- translation ----------
st.markdown("## 1) Translate SAS score code to Python")
col_t1, col_t2 = st.columns([1, 3])
with col_t1:
    can_translate = st.button("Translate SAS → Python", type="primary", use_container_width=True, disabled=not bool(sas_path))
with col_t2:
    st.caption(f"Translates your SAS DATA step score code into a Python function (runs in {ENGINE_LABEL}).")

sas_is_ds2 = False
expected_auto_for_info = []
if can_translate and sas_path:
    raw_code = load_file_text(sas_path)
    sas_is_ds2 = is_ds2_astore(raw_code)
    if sas_is_ds2:
        st.warning("This SAS file looks like DS2/ASTORE (e.g., scoreRecord). Translation is not supported. Use 'Score on Viya (REST)'.")
    else:
        try:
            expected_auto_for_info = extract_expected_inputs_from_sas(raw_code)
            st.session_state.cat_levels = parse_categorical_levels_from_sas(raw_code)
            st.session_state.cat_levels_ci = build_cat_levels_ci(st.session_state.cat_levels)
            # Unión: inputVar.json ∪ requeridos del SAS, excluyendo BAD
            combined_inputs = sorted(set((expected_from_json or [])) | set(expected_auto_for_info or []))
            combined_inputs = [c for c in combined_inputs if c.upper() != "BAD"]
            score_fn, py_code, expected = sct.compile_sas_score(
                raw_code, func_name="sas_score",
                expected_inputs=(combined_inputs or None)
            )
            st.session_state.score_fn = score_fn
            st.session_state.score_py = py_code
            st.session_state.expected_cols = combined_inputs or expected or []
            st.success(f"Translation OK. Inputs on form: {len(st.session_state.expected_cols)} (target BAD excluded).")
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
    row = build_single_record_form(fields, sample_df, st.session_state.cat_levels_ci, key_prefix="gh_single")

    st.markdown(f"## 3) Score in {ENGINE_LABEL}")
    do_local = st.button(f"Score in {ENGINE_LABEL}", type="primary", use_container_width=True, key="btn_gh_single",
                         disabled=(st.session_state.score_fn is None))

    if do_local:
        copy_aliases_inplace_ci(row, st.session_state.cat_levels_ci)
        invalid = []
        for key_ci, levels in st.session_state.cat_levels_ci.items():
            base = key_ci.lstrip("_")
            val = None
            for k in row.keys():
                if k.lower() == key_ci or k.lower() == base:
                    val = row[k]; break
            if levels and val is not None and str(val) not in levels:
                invalid.append(f"{base.upper()}='{val}' (allowed: {', '.join(levels)})")
        if invalid:
            st.error("Invalid categorical values: " + "; ".join(invalid))
        else:
            missing_now = list_missing(fields, row)
            if missing_now:
                st.error("Complete the required inputs before scoring: " + ", ".join(missing_now))
            else:
                try:
                    fields_full = list(fields)
                    cat_keys   = set(st.session_state.cat_levels_ci.keys())
                    force_keys = set(FORCE_CAT)
                    names = cat_keys | set(f"_{f}" for f in force_keys) | force_keys
                    for nm in names:
                        base = nm.lstrip("_")
                        alias = f"_{base}"
                        if base not in fields_full:
                            fields_full.append(base)
                        if alias not in fields_full:
                            fields_full.append(alias)

                    df_in = pd.DataFrame([row]).reindex(columns=fields_full, fill_value=None)

                    cat_fields_ci = {k.lstrip("_").lower() for k in st.session_state.cat_levels_ci.keys()} | set(FORCE_CAT)
                    for col in df_in.columns:
                        if col.lower() not in cat_fields_ci:
                            df_in[col] = pd.to_numeric(df_in[col], errors="coerce").fillna(0.0)

                    raw_out = st.session_state.score_fn(**df_in.iloc[0].to_dict()) or {}
                    if "EM_EVENTPROBABILITY" not in raw_out:
                        scored = sct.score_dataframe(st.session_state.score_fn, df_in, threshold=thr)
                        p_raw = scored["prob_BAD"].iloc[0] if "prob_BAD" in scored else float("nan")
                        lbl_raw = scored["label_BAD"].iloc[0] if "label_BAD" in scored else None
                    else:
                        p_raw = raw_out.get("EM_EVENTPROBABILITY", float("nan"))
                        lbl_raw = raw_out.get("EM_CLASSIFICATION", None)

                    try:
                        p = float(p_raw)
                    except Exception:
                        p = float("nan")
                    lbl = None if (p != p) else (1 if p >= thr else 0)

                    if p != p:
                        st.warning("Probability is NaN. Check categorical values and numeric fields.")
                    else:
                        st.success(f"Scored in {ENGINE_LABEL}.")
                    c1, c2 = st.columns(2)
                    c1.metric(f"Probability of BAD ({ENGINE_LABEL})", f"{p:0.4f}" if p == p else "—")
                    c2.metric(f"Predicted label ({ENGINE_LABEL})", "1" if lbl == 1 else ("0" if lbl == 0 else "—"))
                except Exception as e:
                    st.error(f"{ENGINE_LABEL} scoring failed: {e}")

# ---- Single case — Viya (REST) ----
with tabs[1]:
    st.markdown("## 2) Provide inputs for Viya (REST)")
    fields = st.session_state.expected_cols or list(sample_df.columns)
    if not fields:
        st.info("No input schema found. Add score/inputVar.json or a data/sample.csv.")
    row_rest = build_single_record_form(fields, sample_df, st.session_state.cat_levels_ci, key_prefix="viya_single")

    st.markdown("## 3) Score on Viya (REST)")
    do_rest  = st.button("Score on Viya (REST)", use_container_width=True, key="btn_viya_single")

    if do_rest:
        copy_aliases_inplace_ci(row_rest, st.session_state.cat_levels_ci)
        invalid = []
        for key_ci, levels in st.session_state.cat_levels_ci.items():
            base = key_ci.lstrip("_")
            val = None
            for k in row_rest.keys():
                if k.lower() == key_ci or k.lower() == base:
                    val = row_rest[k]; break
            if levels and val is not None and str(val) not in levels:
                invalid.append(f"{base.upper()}='{val}' (allowed: {', '.join(levels)})")
        if invalid:
            st.error("Invalid categorical values: " + "; ".join(invalid))
        else:
            missing_now = list_missing(fields, row_rest)
            if missing_now:
                st.error("Complete the required inputs before scoring: " + ", ".join(missing_now))
            else:
                try:
                    with st.spinner("Calling Viya..."):
                        resp = score_row_via_rest(row_rest)
                    p, lbl = parse_mas_outputs(resp, threshold=thr)
                    st.success("Viya scoring done.")
                    c1, c2 = st.columns(2)
                    c1.metric("Probability of BAD (Viya)", f"{p:0.4f}" if not pd.isna(p) else "—")
                    c2.metric("Predicted label (Viya)", "1" if lbl == 1 else ("0" if lbl == 0 else "—"))
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

            cat_keys   = set(st.session_state.cat_levels_ci.keys())
            force_keys = set(FORCE_CAT)
            names = cat_keys | set(f"_{f}" for f in force_keys) | force_keys
            for nm in names:
                base = nm.lstrip("_")
                alias = f"_{base}"
                if nm in df.columns and base not in df.columns:
                    df[base] = df[nm]
                if base in df.columns and alias not in df.columns:
                    df[alias] = df[base]

            cat_fields_ci = {k.lstrip("_").lower() for k in st.session_state.cat_levels_ci.keys()} | set(FORCE_CAT)
            for col in df.columns:
                if col.lower() not in cat_fields_ci:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

            c1, c2 = st.columns(2)
            do_local_csv = c1.button(f"Score CSV in {ENGINE_LABEL}",
                                     use_container_width=True, disabled=(st.session_state.score_fn is None), key="btn_gh_csv")
            do_rest_csv  = c2.button("Score CSV on Viya (REST)", use_container_width=True, key="btn_viya_csv")

            if do_local_csv:
                try:
                    scored = sct.score_dataframe(st.session_state.score_fn, df, threshold=thr)
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
    sas_file = find_sas_score_file()
    st.write("- SAS score file:", f"`{sas_file}`" if sas_file else "_not found_")
    st.write("- Inputs from inputVar.json:", len(expected_from_json))
    if expected_auto_for_info:
        with st.expander("Inputs detected from SAS (missing(...))"):
            st.code("\n".join(expected_auto_for_info))
    if st.session_state.cat_levels:
        with st.expander("Categorical levels detected from SAS (select/when + aliases)"):
            st.json(st.session_state.cat_levels)
    st.write("- Current expected inputs (union JSON ∪ SAS, excluding BAD):", len(st.session_state.expected_cols))
    if st.session_state.expected_cols:
        with st.expander("Show expected input fields"):
            st.code("\n".join(st.session_state.expected_cols))
    st.write("- Forced categoricals:", ", ".join(sorted(FORCE_CAT)) or "—")
