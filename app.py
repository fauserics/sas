# app.py
# Real Time Scoring App (powered by SAS)

import os, re, json, math, random
import pandas as pd
import streamlit as st

# LLM refinement wrapper (ya creaste llm/assistant.py)
from llm.assistant import refine_reply_with_llm

# Tolerant translator mode
os.environ["SC_FORCE_FALLBACK"] = "1"

APP_TITLE = "Real Time Scoring App (powered by SAS)"
ENGINE_LABEL = "GitHub (Python)"

# Categorical hints
FORCE_CAT = {"reason"}
DEFAULT_LEVELS = {"reason": ["DebtCon", "HomeImp"]}

SCORE_DIR = "score"
DATA_DIR = "data"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# =========================
# Optional translator import
# =========================
sct = None
translator_import_error = None
try:
    import sas_code_translator as sct  # if missing, use inline fallback
except Exception as e:
    translator_import_error = e

# =========================
# Inline fallback for DATA step logistic
# =========================
def _to_num(x):
    try:
        v = float(x)
        if math.isnan(v): return 0.0
        return v
    except Exception:
        return 0.0

def _discover_levels_and_x(sas_code):
    m = re.search(r"array\s+_beta_[A-Za-z0-9_]+\s*\[\s*(\d+)\s*\]\s*_temporary_\s*\((.*?)\)\s*;", sas_code, flags=re.I|re.S)
    if not m:
        raise RuntimeError("No beta array found in SAS code.")
    n = int(m.group(1))
    toks = [t for t in re.split(r"[,\s]+", m.group(2).strip()) if t]
    if len(toks) != n:
        raise RuntimeError("Beta length mismatch.")
    beta = [None] + [float(t) for t in toks]

    x_assigns = {}
    for i_str, expr in re.findall(r"_xrow_[A-Za-z0-9_]+\s*\[\s*(\d+)\s*\]\s*=\s*([^;]+);", sas_code, flags=re.I):
        x_assigns[int(i_str)] = expr.strip()

    reason_map = {}
    sel_var = None
    sel = re.search(r"select\s*\(\s*([A-Za-z_]\w*)\s*\)\s*;(.+?)end\s*;", sas_code, flags=re.I|re.S)
    if sel:
        sel_var = sel.group(1).strip()
        block = sel.group(2)
        for cond, stmt in re.findall(r"when\s*\(\s*('.*?')\s*\)\s*([^\;]+)\s*;", block, flags=re.I|re.S):
            mset = re.search(r"_xrow_[A-Za-z0-9_]+\s*\[\s*(\d+)\s*\]\s*=\s*_temp_", stmt, flags=re.I)
            if mset:
                reason_map[int(mset.group(1))] = cond.strip()

    if not sel_var:
        sel_var = "_REASON_" if re.search(r"\b_REASO[N_]\b", sas_code, flags=re.I) else ("REASON" if re.search(r"\bREASON\b", sas_code, flags=re.I) else "_REASON_")

    miss = set()
    for v in re.findall(r"missing\(\s*([A-Za-z_][\w']*)\s*\)", sas_code, flags=re.I):
        v = re.sub(r"^'([^']+)'\s*n$", r"\1", v)
        if v.upper() != "BAD":
            miss.add(v)
    if "REASON" in sas_code.upper():
        miss.add("REASON")

    return beta, x_assigns, reason_map, sel_var, sorted(miss)

def _compile_logit_fallback_inline(sas_code, expected_inputs=None):
    beta, x_assigns, reason_map, sel_var, miss = _discover_levels_and_x(sas_code)
    n = len(beta) - 1
    inputs = sorted(set(miss) | set(expected_inputs or []))

    body = []
    for v in inputs:
        body.append(f"{v} = row.get('{v}', None)")
    body.append("_temp_ = 1.0")
    body.append(f"x = [0.0]*({n+1})")

    if 1 in x_assigns and re.fullmatch(r"1(\.0+)?", x_assigns[1]):
        body.append("x[1] = 1.0")

    if reason_map and sel_var:
        for idx in sorted(reason_map):
            body.append(f"x[{idx}] = 1.0 if {sel_var} == {reason_map[idx]} else 0.0")

    for idx in sorted(x_assigns):
        if idx == 1 and re.fullmatch(r"1(\.0+)?", x_assigns[1]):
            continue
        if idx in reason_map:
            continue
        expr = x_assigns[idx].strip()
        if re.fullmatch(r"\d+(\.\d+)?", expr):
            body.append(f"x[{idx}] = float({expr})")
        else:
            body.append("try:")
            body.append(f"    x[{idx}] = _to_num({expr})")
            body.append("except Exception:")
            body.append(f"    x[{idx}] = 0.0")

    body.append(f"BETA = {beta}")
    body.append(f"_linp_ = sum(x[i]*BETA[i] for i in range(1, {n+1}))")
    body.append("if not math.isfinite(_linp_): _linp_ = 0.0")
    body.append("p1 = (1.0/(1.0+math.exp(-_linp_)) if _linp_>0 else math.exp(_linp_)/(1.0+math.exp(_linp_)))")
    body.append("return {'EM_EVENTPROBABILITY': float(p1)}")

    src = "def __scorer__(**row):\n" + "\n".join("    "+ln for ln in body)
    loc = {"math": math, "_to_num": _to_num}
    exec(src, loc, loc)
    scorer = loc["__scorer__"]

    py_display = "def sas_score(**row):\n" + "\n".join("    "+ln for ln in body)
    return scorer, py_display, inputs

def _score_dataframe_inline(score_fn, df, threshold=0.5):
    rows = []
    for _, r in df.iterrows():
        out = score_fn(**r.to_dict()) or {}
        p = out.get("EM_EVENTPROBABILITY", float("nan"))
        try:
            p = float(p)
        except Exception:
            p = float("nan")
        lbl = None if (p!=p) else (1 if p >= threshold else 0)
        rec = r.to_dict()
        rec["prob_BAD"] = p
        rec["label_BAD"] = (int(lbl) if lbl is not None else None)
        rows.append(rec)
    return pd.DataFrame(rows)

# Use fallback if translator missing
if sct is None:
    def compile_sas_score(code, func_name="sas_score", expected_inputs=None):
        return _compile_logit_fallback_inline(code, expected_inputs=expected_inputs)
    def score_df_local(fn, df, threshold=0.5):
        return _score_dataframe_inline(fn, df, threshold=threshold)
    inline_translator_active = True
else:
    compile_sas_score = sct.compile_sas_score
    score_df_local = sct.score_dataframe
    inline_translator_active = False

# =========================
# Viya REST client
# =========================
try:
    from viya_mas_client import score_row_via_rest
except Exception:
    def score_row_via_rest(_row):
        raise RuntimeError("viya_mas_client not available. Set VIYA_URL/MAS_MODULE_ID or add the module.")

# =========================
# App helpers
# =========================
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
        if isinstance(data, list):
            if data and isinstance(data[0], dict) and "name" in data[0]:
                return [d["name"] for d in data]
            return [str(x) for x in data]
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
    if m: v = m.group(1)
    v = re.sub(r"\W", "_", v)
    return v

def extract_expected_inputs_from_sas(code):
    miss_names = set(_sanitize_var(x) for x in re.findall(r"missing\(\s*([A-Za-z_][\w']*)\s*\)", code, flags=re.I))
    if re.search(r"\bREASON\b", code, flags=re.I) or re.search(r"\b_REASO[N_]\b", code, flags=re.I):
        miss_names.add("REASON")
    drop_prefixes = ("_", "EM_", "P_", "I_", "va__d__E_")
    keep = [n for n in miss_names if n and not n.upper().startswith(drop_prefixes)]
    return sorted(keep, key=str.upper)

def parse_categorical_levels_from_sas(code):
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
    ci = {}
    for k, v in cat_levels.items():
        ci[k.lower()] = v
        base = k.lstrip("_").lower()
        ci[base] = v
        ci["_"+base] = v
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

def build_single_record_form(fields, sample, cat_levels_ci, key_prefix):
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
    names = set()
    for k in list(cat_levels_ci.keys()):
        base = k.lstrip("_")
        names.add(base.lower()); names.add("_"+base.lower())
    for f in FORCE_CAT:
        names.add(f); names.add("_"+f)
    def get_ci(d, key):
        for kk in d.keys():
            if kk.lower() == key.lower(): return d[kk], kk
        return None, None
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
    try: p = None if prob is None else float(prob)
    except Exception: p = None
    try:
        lbl = None if label is None else (int(label) if str(label).isdigit() else (1 if (p is not None and p >= threshold) else 0))
    except Exception:
        lbl = (1 if (p is not None and p >= threshold) else 0)
    return (p if p is not None else float("nan")), lbl

# =========================
# Institutional assistant helpers
# =========================
def _band_from_prob(prob, threshold):
    if prob is None or (isinstance(prob, float) and (prob != prob)):
        return "unknown"
    p = float(prob)
    if p < threshold * 0.6:            return "very_low"
    if p < threshold:                  return "near_low"
    if p < min(0.85, threshold + 0.2): return "elevated"
    return "very_high"

def _institutional_template(prob, threshold, row=None):
    """Deterministic reply approved by the institution, one per band."""
    row = row or {}
    reason = (row.get("REASON") or row.get("_REASON_") or "").strip()
    loan = row.get("LOAN")
    amount_txt = ""
    try:
        if loan is not None and str(loan).strip() != "":
            amount_txt = f" of {int(float(loan)):,}"
    except Exception:
        amount_txt = ""
    reason_txt = f" for **{reason}**" if reason else ""

    band = _band_from_prob(prob, threshold)

    TEMPLATES_FIXED = {
        "unknown": (
            "technical",
            "I couldn’t compute a reliable probability right now. Could you confirm a few details"
            f"{amount_txt} (e.g., Reason and Employment years)? I’ll recheck immediately."
        ),
        "very_low": (
            "approve_low",
            "Thanks for the details! Based on your profile{reason}, you look **well positioned** for approval. "
            "I can proceed with the next steps (document upload and a quick ID check). Shall I continue?"
        ),
        "near_low": (
            "approve_borderline",
            "You’re **close to our approval threshold**. We can submit as is, or strengthen the case "
            f"(smaller amount{amount_txt}, extra documents). Which do you prefer?"
        ),
        "elevated": (
            "cautionary",
            "Your profile suggests **higher risk** right now. We can explore alternatives: a smaller amount"
            f"{amount_txt}, a longer term, or a guarantor. Would you like to try one together?"
        ),
        "very_high": (
            "decline_empatic",
            "Thank you for your time. I know this isn’t the outcome you hoped for. We’re **unable to proceed** today. "
            "If you’d like, I can share practical steps to strengthen your profile and when to reapply."
        ),
    }

    tone, base = TEMPLATES_FIXED.get(band, TEMPLATES_FIXED["unknown"])
    base = base.replace("{reason}", reason_txt)
    return tone, base

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("### Settings")
    thr = st.slider("Decision threshold (BAD=1)", 0.0, 1.0, 0.50, 0.01, key="thr")
    debug = st.toggle("Debug mode", value=False, key="debug")

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

# =========================
# Load base files
# =========================
expected_from_json = load_input_vars()
sas_path = find_sas_score_file()
if not sas_path:
    st.caption("No SAS score file found under ./score. Place your exported DATA step (*.sas) there.")
else:
    st.caption(f"Using SAS score file: `{sas_path}`")
sample_df = load_sample_df(expected_from_json)

# =========================
# Translator diagnostics (optional)
# =========================
if translator_import_error:
    st.markdown("### Translator import error (sas_code_translator.py)")
    st.error(repr(translator_import_error))
    try:
        code_txt = open("sas_code_translator.py","r",encoding="utf-8").read()
        try:
            compile(code_txt, "sas_code_translator.py", "exec")
        except SyntaxError as se:
            st.warning(f"SyntaxError at line {se.lineno}, col {se.offset}: {se.msg}")
            lines = code_txt.splitlines()
            start = max(0, (se.lineno or 1)-4); end = min(len(lines), (se.lineno or 1)+3)
            snippet = "\n".join(f"{i+1:>4}: {lines[i]}" for i in range(start, end))
            st.code(snippet, language="python")
    except Exception as ex:
        st.info(f"Could not open/parse sas_code_translator.py: {ex}")

# =========================
# SAS → Python translation (GitHub engine)
# =========================
st.markdown("## 1) Translate SAS score code to Python")
col_t1, col_t2 = st.columns([1, 3])
with col_t1:
    can_translate = st.button("Translate SAS → Python", type="primary", use_container_width=True, disabled=not bool(sas_path), key="btn_translate")
with col_t2:
    src_engine = "INLINE fallback" if sct is None else "sas_code_translator.py"
    st.caption(f"Translator engine: {src_engine}. Runs the Python scorer in {ENGINE_LABEL}.")

if "score_fn" not in st.session_state: st.session_state.score_fn = None
if "score_py" not in st.session_state: st.session_state.score_py = ""
if "expected_cols" not in st.session_state: st.session_state.expected_cols = expected_from_json
if "cat_levels" not in st.session_state: st.session_state.cat_levels = {}
if "cat_levels_ci" not in st.session_state: st.session_state.cat_levels_ci = {}

if can_translate and sas_path:
    raw_code = load_file_text(sas_path)
    try:
        expected_auto = extract_expected_inputs_from_sas(raw_code)
        st.session_state.cat_levels = parse_categorical_levels_from_sas(raw_code)
        st.session_state.cat_levels_ci = build_cat_levels_ci(st.session_state.cat_levels)
        # ---- ÚNICO CAMBIO: forzar set en ambos lados antes del operador | ----
        combined_inputs = sorted(set(expected_from_json or []) | set(expected_auto or []))
        combined_inputs = [c for c in combined_inputs if c.upper() != "BAD"]
        score_fn, py_code, expected = compile_sas_score(
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

# =========================
# Tabs
# =========================
tabs = st.tabs([f"Single case — {ENGINE_LABEL}", "Single case — Viya (REST)", "CSV batch", "Assistant", "Info"])

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
                        scored = score_df_local(st.session_state.score_fn, df_in, threshold=st.session_state["thr"])
                        p_raw = scored["prob_BAD"].iloc[0] if "prob_BAD" in scored else float("nan")
                    else:
                        p_raw = raw_out.get("EM_EVENTPROBABILITY", float("nan"))

                    try:
                        p = float(p_raw)
                    except Exception:
                        p = float("nan")
                    lbl = None if (p != p) else (1 if p >= st.session_state["thr"] else 0)

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
                    p, lbl = parse_mas_outputs(resp, threshold=st.session_state["thr"])
                    st.success("Viya scoring done.")
                    c1, c2 = st.columns(2)
                    c1.metric("Probability of BAD (Viya)", f"{p:0.4f}" if not pd.isna(p) else "—")
                    c2.metric("Predicted label (Viya)", "1" if lbl == 1 else ("0" if lbl == 0 else "—"))
                    if st.session_state["debug"]:
                        st.subheader("Debug — Viya raw response")
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
                    scored = score_df_local(st.session_state.score_fn, df, threshold=st.session_state["thr"])
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
                        p, lbl = parse_mas_outputs(resp, threshold=st.session_state["thr"])
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

# ---- Assistant (Conversational, English) ----
with tabs[3]:
    st.markdown("## Conversational Assistant (English)")
    st.caption("Choose institutional reply only, or institutional reply refined by an LLM. No probabilities are disclosed to customers.")

    eng = st.radio("Engine", [ENGINE_LABEL, "Viya (REST)"], horizontal=True, key="asst_engine")

    # Clear mode: institutional vs institutional + LLM
    resp_mode = st.radio(
        "Response mode",
        ["Institutional (template only)", "Institutional + LLM"],
        index=0,
        horizontal=True,
        key="resp_mode"
    )
    use_llm = (resp_mode == "Institutional + LLM")
    if use_llm:
        col_llm1, col_llm2 = st.columns([1,1])
        llm_temp = col_llm1.slider("LLM temperature", 0.0, 1.0, 0.6, 0.05, key="assistant_llm_temp")
        col_llm2.caption(f"Model: {os.getenv('LLM_MODEL', 'gpt-4o-mini')} (set LLM_MODEL env var to change)")

    fields = st.session_state.expected_cols or list(sample_df.columns)
    if not fields:
        st.info("No input schema found. Add score/inputVar.json or a data/sample.csv.")
    row_asst = build_single_record_form(fields, sample_df, st.session_state.cat_levels_ci, key_prefix="assistant_single")

    def _score_one(row_dict):
        copy_aliases_inplace_ci(row_dict, st.session_state.cat_levels_ci)
        missing_now = list_missing(fields, row_dict)
        if missing_now:
            return None, f"Missing required inputs: {', '.join(missing_now)}"

        fields_full = list(fields)
        names = set(st.session_state.cat_levels_ci.keys()) | {f'_{f}' for f in FORCE_CAT} | set(FORCE_CAT)
        for nm in names:
            base = nm.lstrip("_")
            alias = f"_{base}"
            if base not in fields_full: fields_full.append(base)
            if alias not in fields_full: fields_full.append(alias)

        df_in = pd.DataFrame([row_dict]).reindex(columns=fields_full, fill_value=None)
        cat_fields_ci = {k.lstrip("_").lower() for k in st.session_state.cat_levels_ci.keys()} | set(FORCE_CAT)
        for col in df_in.columns:
            if col.lower() not in cat_fields_ci:
                df_in[col] = pd.to_numeric(df_in[col], errors="coerce").fillna(0.0)

        if eng == ENGINE_LABEL:
            if st.session_state.score_fn is None:
                return None, "Translate SAS → Python first in step 1."
            out = st.session_state.score_fn(**df_in.iloc[0].to_dict()) or {}
            p = out.get("EM_EVENTPROBABILITY")
            if p is None:
                scored = score_df_local(st.session_state.score_fn, df_in, threshold=st.session_state["thr"])
                p = scored["prob_BAD"].iloc[0]
            try:
                p = float(p)
            except Exception:
                p = float("nan")
            return p, None
        else:
            try:
                resp = score_row_via_rest(row_dict)
                p, _ = parse_mas_outputs(resp, threshold=st.session_state["thr"])
                return float(p), None
            except Exception as e:
                return None, f"Viya REST error: {e}"

    st.markdown("### Run assistant")
    if st.button("Get suggested reply", type="primary", use_container_width=True, key="assistant_btn"):
        prob, err = _score_one(dict(row_asst))
        st.chat_message("assistant").markdown("I'll review your inputs and craft a helpful, empathetic reply.")
        if err:
            st.chat_message("assistant").markdown(f"**Issue:** {err}")
        else:
            # 1) Institutional deterministic reply
            tone, base_reply = _institutional_template(prob, st.session_state["thr"], row_asst)

            # 2) Optional LLM refinement
            final_reply = base_reply
            dbg = ""
            if use_llm:
                final_reply, dbg = refine_reply_with_llm(
                    base_reply=base_reply,
                    prob=prob,
                    threshold=st.session_state["thr"],
                    row=row_asst,
                    temperature=llm_temp if 'llm_temp' in locals() else 0.6,
                    model=os.getenv("LLM_MODEL", "gpt-4o-mini")
                )

            pretty_p = "—" if prob is None or pd.isna(prob) else f"{prob:0.4f}"

            st.chat_message("assistant").markdown(
                f"**Predicted probability (BAD):** {pretty_p}\n\n"
                f"**Institutional (template):**\n\n{base_reply}\n\n"
                + (f"**Refined (LLM):**\n\n{final_reply}\n\n" if use_llm else "")
                + f"_Tone: {tone}; Threshold: {st.session_state['thr']:0.2f}_"
            )
            if st.session_state.get("debug") and use_llm:
                st.caption(f"LLM debug: {dbg}")

# ---- Info ----
with tabs[4]:
    st.markdown("## Info")
    sas_file = find_sas_score_file()
    st.write("- SAS score file:", f"`{sas_file}`" if sas_file else "_not found_")
    st.write("- Inputs from inputVar.json:", len(expected_from_json))
    st.write("- Translator engine:", "INLINE fallback" if sct is None else "sas_code_translator.py")
    if st.session_state.expected_cols:
        with st.expander("Expected input fields"):
            st.code("\n".join(st.session_state.expected_cols))
