# app.py
# Real Time Scoring App (powered by SAS)

import os
import re
import json
import pandas as pd
import streamlit as st

# Fuerzo fallback del traductor para evitar NaN/None
os.environ["SC_FORCE_FALLBACK"] = "1"

from sas_code_translator import compile_sas_score, score_dataframe as score_df_local
from viya_mas_client import score_row_via_rest  # requiere VIYA_URL, MAS_MODULE_ID y auth

APP_TITLE = "Real Time Scoring App (powered by SAS)"
ENGINE_LABEL = "GitHub (Python)"

FORCE_CAT = {"reason"}
DEFAULT_LEVELS = {"reason": ["DebtCon", "HomeImp"]}

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

SCORE_DIR = "score"
DATA_DIR = "data"

# ---------- helpers (igual que antes; omitidos para brevedad) ----------
# ... (usa exactamente los mismos helpers de la última versión que ya tenés)
# Asegurate de conservar: find_sas_score_file, load_input_vars, extract_expected_inputs_from_sas,
# parse_categorical_levels_from_sas, build_cat_levels_ci, load_sample_df, parse_mas_outputs,
# build_single_record_form, is_ds2_astore, list_missing, copy_aliases_inplace_ci

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

# ---------- carga y translate (igual que antes) ----------
# ... (misma lógica de: expected_from_json, sas_path, sample_df, session_state, Translate button, etc.)

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
        # validar categóricas
        invalid = []
        for key_ci, levels in st.session_state.cat_levels_ci.items():
            base = key_ci.lstrip("_")
            val = next((row[k] for k in row if k.lower() in (key_ci, base)), None)
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
                    # Ensanchar con alias
                    fields_full = list(fields)
                    cat_keys   = set(st.session_state.cat_levels_ci.keys())
                    force_keys = set(FORCE_CAT)
                    names = cat_keys | {f"_{f}" for f in force_keys} | force_keys
                    for nm in names:
                        base = nm.lstrip("_"); alias = f"_{base}"
                        if base not in fields_full: fields_full.append(base)
                        if alias not in fields_full: fields_full.append(alias)

                    df_in = pd.DataFrame([row]).reindex(columns=fields_full, fill_value=None)

                    # Cast numérico y rellenar NaN con 0.0 para no contaminar _linp_
                    cat_fields_ci = {k.lstrip("_").lower() for k in st.session_state.cat_levels_ci.keys()} | set(FORCE_CAT)
                    for col in df_in.columns:
                        if col.lower() not in cat_fields_ci:
                            df_in[col] = pd.to_numeric(df_in[col], errors="coerce").fillna(0.0)

                    # DEBUG: mostrar entrada que se usará
                    if debug:
                        st.subheader("Debug — input row used")
                        st.json(df_in.iloc[0].to_dict())

                    # Score directo con dict para inspeccionar raw_out
                    raw_out = st.session_state.score_fn(**df_in.iloc[0].to_dict()) or {}
                    if debug:
                        st.subheader("Debug — raw scorer output")
                        st.json(raw_out)

                    # Si viene vacío, probamos helper por DataFrame
                    if "EM_EVENTPROBABILITY" not in raw_out:
                        scored = score_df_local(st.session_state.score_fn, df_in, threshold=thr)
                        p_raw = scored["prob_BAD"].iloc[0] if "prob_BAD" in scored else float("nan")
                        lbl_raw = scored["label_BAD"].iloc[0] if "label_BAD" in scored else None
                    else:
                        p_raw = raw_out.get("EM_EVENTPROBABILITY", float("nan"))
                        lbl_raw = raw_out.get("EM_CLASSIFICATION", None)

                    try:
                        p = float(p_raw)
                    except Exception:
                        p = float("nan")
                    lbl = None if (p!=p) else (1 if p >= thr else 0)

                    if p!=p:
                        st.warning("Probability is NaN. Check categorical values and numeric fields.")
                    else:
                        st.success(f"Scored in {ENGINE_LABEL}.")
                    c1, c2 = st.columns(2)
                    c1.metric(f"Probability of BAD ({ENGINE_LABEL})", f"{p:0.4f}" if p==p else "—")
                    c2.metric(f"Predicted label ({ENGINE_LABEL})", "1" if lbl == 1 else ("0" if lbl == 0 else "—"))
                except Exception as e:
                    st.error(f"{ENGINE_LABEL} scoring failed: {e}")

# ---- Single case — Viya (REST) ----
# (igual que antes)

# ---- CSV batch ----
# (igual que antes, pero mantené el cast .fillna(0.0) para numéricos)

# ---- Info ----
# (igual que antes)
