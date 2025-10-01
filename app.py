# app.py — Real Time Scoring App (powered by SAS)

import os
import json
import io
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd

# ============================================================
# Set up page
# ============================================================
st.set_page_config(page_title="Real Time Scoring App (powered by SAS)", layout="wide")

# ============================================================
# Load secrets into environment (override)
# ============================================================
try:
    for k, v in st.secrets.items():
        os.environ[k] = str(v)
except Exception:
    pass

APP_TITLE = "Real Time Scoring App (powered by SAS)"
st.title(APP_TITLE)

# ============================================================
# Safe helpers for category levels merging (fix list|set)
# ============================================================
def _as_set(x):
    if x is None:
        return set()
    if isinstance(x, set):
        return x
    if isinstance(x, dict):
        # a veces vienen como {"levels":[...]} o dict raro
        if "levels" in x and isinstance(x["levels"], (list, set, tuple)):
            return set(x["levels"])
        return set(x.keys())
    if isinstance(x, (list, tuple)):
        return set(x)
    return {x}

def _merge_levels(a, b):
    return sorted(list(_as_set(a) | _as_set(b)))

# ============================================================
# Optional imports with graceful fallback
# ============================================================
# SAS score translator (local Python scoring)
try:
    import sas_code_translator as sct
except Exception as e:
    sct = None
    st.warning(f"Translator module not found/failed to import: {e}")

# Viya MAS client (REST scoring)
try:
    from viya_mas_client import score_row_via_rest, get_module_inputs, sync_input_schema_to_file
except Exception as e:
    score_row_via_rest = None
    get_module_inputs = None
    sync_input_schema_to_file = None
    st.warning(f"Viya client not found/failed to import: {e}")

# Optional LLM assistant
try:
    from llm.assistant import suggest_message, refine_with_llm
except Exception:
    suggest_message = None
    refine_with_llm = None

# ============================================================
# Load input variables schema
# ============================================================
def load_input_vars(path: str = "score/inputVar.json") -> List[Dict[str, Any]]:
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                j = json.load(f)
            items = j.get("inputVariables") or j.get("variables") or []
            fields = []
            for it in items:
                name = it.get("name")
                if not name:
                    continue
                typ = (it.get("type") or it.get("role") or "").lower()
                levels = it.get("levels")
                fields.append({"name": name, "type": typ, "levels": levels})
            return fields
        except Exception as e:
            st.error(f"Failed to read score/inputVar.json: {e}")
    # Default (minimal) if file missing:
    return [
        {"name":"REASON","type":"cat","levels":["DebtCon","HomeImp"]},
        {"name":"CLAGE","type":"num"},
        {"name":"CLNO","type":"num"},
        {"name":"DEBTINC","type":"num"},
        {"name":"DELINQ","type":"num"},
        {"name":"DEROG","type":"num"},
        {"name":"LOAN","type":"num"},
        {"name":"MORTDUE","type":"num"},
        {"name":"NINQ","type":"num"},
        {"name":"VALUE","type":"num"},
        {"name":"YOJ","type":"num"},
    ]

FIELDS = load_input_vars()

# Keep UI category overrides in session
if "ui_categories" not in st.session_state:
    st.session_state["ui_categories"] = {f["name"]: f.get("levels") for f in FIELDS if f.get("levels")}
if "score_fn" not in st.session_state:
    st.session_state["score_fn"] = None
if "last_row" not in st.session_state:
    st.session_state["last_row"] = None
if "last_prob" not in st.session_state:
    st.session_state["last_prob"] = None

# ============================================================
# Form builder for single record
# ============================================================
def build_single_record_form(fields: List[Dict[str, Any]], sample: Optional[pd.DataFrame]=None,
                             key_prefix: str = "form") -> Dict[str, Any]:
    """
    Build a single-record input form. Uses st.session_state['ui_categories'] to decide categorical options.
    key_prefix ensures unique widget keys across tabs/forms.
    """
    ui_cats = st.session_state.get("ui_categories") or {}
    row: Dict[str, Any] = {}
    # layout in 2 columns
    cols = st.columns(2)
    col_idx = 0

    for f in fields:
        c = f["name"]
        cat_levels = ui_cats.get(c) or f.get("levels")

        tgt = cols[col_idx % 2]
        col_idx += 1

        # Infer default value from sample (if provided)
        default_val = None
        if sample is not None and c in sample.columns and len(sample) > 0:
            default_val = sample.iloc[0][c]

        if cat_levels:
            # categorical
            options = list(cat_levels)
            # ensure safe default
            def_index = 0
            if default_val in options:
                def_index = options.index(default_val)
            row[c] = tgt.selectbox(
                c, options, index=def_index, key=f"{key_prefix}_sel_{c}"
            )
        else:
            # numeric/text → assume float unless told otherwise
            try:
                start_val = float(default_val) if default_val is not None and str(default_val) != "" else 0.0
            except Exception:
                start_val = 0.0
            row[c] = tgt.number_input(
                c, value=float(start_val), key=f"{key_prefix}_num_{c}"
            )
    return row

# ============================================================
# Scoring helpers
# ============================================================
def score_local_python(row: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
    """Score using the local Python-translated function (if available)."""
    if sct is None or st.session_state.get("score_fn") is None:
        raise RuntimeError("No local Python score function available. Translate SAS → Python first.")
    df = pd.DataFrame([row])
    df_out = sct.score_dataframe(st.session_state["score_fn"], df, threshold=threshold)
    prob = float(df_out["Probability"].iloc[0])
    pred = int(df_out["Predicted"].iloc[0]) if pd.notna(prob) else None
    return {"Probability": prob, "Predicted": pred}

def score_via_viya(row: Dict[str, Any]) -> Dict[str, Any]:
    if score_row_via_rest is None:
        raise RuntimeError("Viya client not available.")
    resp = score_row_via_rest(row)
    # Try common keys
    outputs = {}
    try:
        # MAS returns {"outputs":[{"name":"...","value":...},...]}
        outs = resp.get("outputs") or []
        for item in outs:
            outputs[item.get("name")] = item.get("value")
    except Exception:
        pass
    # Heuristic for probability keys
    prob = None
    for k in ("EM_EVENTPROBABILITY","P_BAD","P_TARGET1","P_bad1","P_va__d__E_JOB1","Probability"):
        if k in outputs and outputs[k] not in (None, ""):
            try:
                prob = float(outputs[k])
                break
            except Exception:
                pass
    return {"raw": resp, "outputs": outputs, "Probability": prob}

# ============================================================
# UI Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Single case — GitHub (Python)",
    "Single case — Viya (REST)",
    "Batch scoring (CSV)",
    "Translate SAS → Python",
    "Assistant"
])

# ------------------------------------------------------------
# TAB 1 — Single case: GitHub (Python)
# ------------------------------------------------------------
with tab1:
    st.subheader("Single case — GitHub (Python)")
    thr = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01, key="thr_py")
    row = build_single_record_form(FIELDS, None, key_prefix="gh")
    if st.button("Score on GitHub (Python)", key="btn_score_py"):
        try:
            out = score_local_python(row, threshold=thr)
            st.success(f"Probability of BAD (GitHub): {out['Probability']:.4f}")
            st.write(f"Predicted label (GitHub): **{out['Predicted']}**")
            st.session_state["last_row"] = row
            st.session_state["last_prob"] = out["Probability"]
        except Exception as e:
            st.error(f"GitHub (Python) scoring failed: {e}")

# ------------------------------------------------------------
# TAB 2 — Single case: Viya (REST)
# ------------------------------------------------------------
with tab2:
    st.subheader("Single case — Viya (REST)")
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Sync form from MAS inputs", key="btn_sync_schema") and sync_input_schema_to_file:
            try:
                names = sync_input_schema_to_file(module_id=os.getenv("MAS_MODULE_ID"))
                st.success(f"Synchronized inputVar.json from MAS: {', '.join(names)}")
                # reload fields and categories
                global FIELDS
                FIELDS = load_input_vars()
                st.session_state["ui_categories"] = {f["name"]: f.get("levels") for f in FIELDS if f.get("levels")}
            except Exception as e:
                st.error(f"Sync failed: {e}")
    with colB:
        st.caption(f"Module ID: `{os.getenv('MAS_MODULE_ID','(not set)')}`")

    row_rest = build_single_record_form(FIELDS, None, key_prefix="viya")
    if st.button("Score on Viya (REST)", key="btn_score_viya"):
        try:
            out = score_via_viya(row_rest)
            prob = out.get("Probability")
            if prob is None or pd.isna(prob):
                st.warning("Probability is NaN (check inputs and categorical values).")
            else:
                st.success(f"Probability of BAD (Viya): {prob:.4f}")
            st.write("Raw outputs:", out.get("outputs"))
            st.session_state["last_row"] = row_rest
            st.session_state["last_prob"] = prob
        except Exception as e:
            st.error(f"Viya REST error: {e}")

# ------------------------------------------------------------
# TAB 3 — Batch scoring (CSV)
# ------------------------------------------------------------
with tab3:
    st.subheader("Batch scoring (CSV)")
    eng = st.selectbox("Engine", ["GitHub (Python)", "Viya (REST)"], key="batch_engine")
    thr_b = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01, key="thr_batch")
    up = st.file_uploader("Upload CSV with columns matching the form", type=["csv"], key="csv_up")
    if st.button("Run batch scoring", key="btn_batch"):
        if up is None:
            st.error("Please upload a CSV.")
        else:
            try:
                df_in = pd.read_csv(io.BytesIO(up.read()))
                if eng == "GitHub (Python)":
                    if st.session_state.get("score_fn") is None:
                        st.error("Translate SAS → Python first to enable GitHub (Python) scoring.")
                    else:
                        df_out = sct.score_dataframe(st.session_state["score_fn"], df_in, threshold=thr_b)
                        st.success("Batch scored with GitHub (Python).")
                        st.dataframe(df_out)
                else:
                    # Viya REST (row-wise)
                    if score_row_via_rest is None:
                        st.error("Viya client not available.")
                    else:
                        rows = []
                        prog = st.progress(0)
                        data = df_in.to_dict(orient="records")
                        for i, r in enumerate(data, start=1):
                            try:
                                out = score_via_viya(r)
                                rows.append({"Probability": out.get("Probability"), **r})
                            except Exception as ex:
                                rows.append({"Probability": None, **r})
                            prog.progress(int(i * 100 / max(1, len(data))))
                        st.success("Batch scored with Viya (REST).")
                        st.dataframe(pd.DataFrame(rows))
            except Exception as e:
                st.error(f"Batch scoring failed: {e}")

# ------------------------------------------------------------
# TAB 4 — Translate SAS → Python
# ------------------------------------------------------------
with tab4:
    st.subheader("Translate SAS → Python")
    st.caption("Paste SAS scoring DATA step (or DS2) below. The translator extracts probabilities if present and category levels where possible.")
    sas_code = st.text_area("SAS score code", height=220, key="sas_code")
    if st.button("Translate SAS → Python", key="btn_translate"):
        if not sct:
            st.error("Translator module not available.")
        else:
            try:
                res = sct.compile_sas_score(sas_code, known_categories=None)
                st.session_state["score_fn"] = res.get("score_fn")
                st.session_state["default_prob"] = res.get("default_prob")

                # Merge categories from form schema and parsed categories (SAFE union)
                parsed_cats = res.get("categories", {}) or {}
                form_levels = {f["name"]: f.get("levels") for f in FIELDS if f.get("levels")}
                all_vars = set(form_levels.keys()) | set(parsed_cats.keys())
                ui_categories = {
                    v: _merge_levels(form_levels.get(v), parsed_cats.get(v))
                    for v in all_vars
                }
                # Keep also any existing ui_categories for other fields
                for k, old in (st.session_state.get("ui_categories") or {}).items():
                    if k not in ui_categories:
                        ui_categories[k] = old

                st.session_state["ui_categories"] = ui_categories

                st.success("Translation OK. Local Python scoring function is ready.")
                if ui_categories:
                    st.write("Detected/merged categorical levels:")
                    st.json(ui_categories)
                if res.get("default_prob") is not None:
                    st.info(f"Default probability hint: {res.get('default_prob')}")
            except Exception as e:
                st.error(f"Translation failed: {e}")

# ------------------------------------------------------------
# TAB 5 — Assistant
# ------------------------------------------------------------
with tab5:
    st.subheader("Assistant")
    st.caption("Respond to customers using institutional templates or refine with an LLM. The model probability can come from GitHub (Python) or Viya (REST).")

    engine = st.selectbox("Score source", ["Last result (any)", "GitHub (Python)", "Viya (REST)"], key="as_engine")
    mode = st.selectbox("Response mode", ["Institutional (deterministic)", "Institutional + LLM (adaptive)"], key="as_mode")

    # Get probability for this session
    prob = st.session_state.get("last_prob")
    row_for_msg = st.session_state.get("last_row") or {}

    # Allow manual override if none
    if prob is None or pd.isna(prob):
        st.info("No probability in session. Enter a probability manually or score a case first.")
        prob = st.number_input("Manual probability (0..1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="as_prob_manual")

    # Base institutional suggestion
    def _institutional(prob_: float) -> str:
        if prob_ >= 0.8:
            return ("We appreciate your interest. Based on our assessment, your application currently does not meet our "
                    "lending criteria. If you'd like, we can share steps to improve your eligibility over time.")
        elif prob_ >= 0.5:
            return ("Thanks for your application. We have a few additional questions to complete our assessment. "
                    "Please confirm your employment details and recent income documentation.")
        elif prob_ >= 0.2:
            return ("Good news — you’re likely eligible. We’ll proceed with standard verification. "
                    "Expect an update from us shortly.")
        else:
            return ("Excellent news — your profile appears very strong. We can move forward with a streamlined approval process.")

    base_msg = _institutional(float(prob))
    st.write("**Institutional suggestion**:")
    st.write(base_msg)

    final_msg = base_msg
    if mode == "Institutional + LLM (adaptive)":
        if refine_with_llm is None:
            st.warning("LLM refinement not available (assistant module not found).")
        else:
            # Provide context to LLM
            ctx = {
                "probability": float(prob),
                "inputs": row_for_msg
            }
            try:
                final_msg = refine_with_llm(base_msg, context=ctx)
            except Exception as e:
                st.warning(f"LLM refinement failed: {e}")
                final_msg = base_msg

    st.write("**Customer-facing message:**")
    st.success(final_msg)

# ============================================================
# Footer / Small diagnostics (optional)
# ============================================================
with st.expander("Session / env (debug)", expanded=False):
    st.write({
        "MAS_MODULE_ID": os.getenv("MAS_MODULE_ID"),
        "VIYA_URL": os.getenv("VIYA_URL"),
        "Have score_fn?": st.session_state.get("score_fn") is not None,
        "UI categories keys": list((st.session_state.get("ui_categories") or {}).keys())[:10]
    })
