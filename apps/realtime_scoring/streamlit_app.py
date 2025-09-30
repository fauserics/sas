import os
import json
import pickle
from io import BytesIO, StringIO
from typing import Dict, Any, List, Tuple

import pandas as pd
import streamlit as st
import requests

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Real Time Scoring App (powered by SAS)", page_icon="⚡", layout="wide")

OWNER = os.environ.get("GH_OWNER", "fauserics")
REPO  = os.environ.get("GH_REPO",  "sas")
BRANCH = os.environ.get("GH_BRANCH", "main")

# model artifacts in the SAME repo (public)
PIPELINE_URL = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/sas/models/pipeline.pkl"
METADATA_URL = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/sas/models/metadata.json"

# -----------------------------
# Helpers
# -----------------------------
def _fetch_bytes(url: str, timeout: int = 20) -> bytes:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "streamlit-app"})
    r.raise_for_status()
    return r.content

def _fetch_json(url: str, timeout: int = 20) -> Dict[str, Any]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "streamlit-app"})
    r.raise_for_status()
    return r.json()

@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[Any, Dict[str, Any]]:
    # metadata first (for threshold & inputs)
    meta = _fetch_json(METADATA_URL)
    raw = _fetch_bytes(PIPELINE_URL)
    try:
        import joblib
        model = joblib.load(BytesIO(raw))
    except Exception:
        model = pickle.load(BytesIO(raw))
    return model, meta

def predict_df(model, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Returns (prob_1, label) using threshold from state/meta."""
    prob = pd.Series(model.predict_proba(df)[:, 1], index=df.index, name="prob_1")
    thr = float(st.session_state.threshold)
    label = (prob >= thr).astype(int).rename("label")
    return prob, label

def build_empty_row(inputs_spec: List[Dict[str, Any]]) -> Dict[str, Any]:
    row = {}
    for x in inputs_spec:
        nm = x.get("name")
        typ = (x.get("type") or "").lower()
        row[nm] = 0.0 if typ == "number" else ""
    return row

# -----------------------------
# Load model + metadata
# -----------------------------
with st.spinner("Loading model & metadata from GitHub..."):
    try:
        model, meta = load_artifacts()
        st.session_state.model_name = meta.get("model_name", "model")
        st.session_state.threshold = meta.get("threshold", 0.5)
        st.session_state.inputs_spec = meta.get("inputs", [])
    except Exception as e:
        st.error(f"Failed to load model or metadata from GitHub.\n{e}")
        st.stop()

# -----------------------------
# Sidebar (Model info)
# -----------------------------
st.sidebar.title("Model")
st.sidebar.write(f"**Name:** `{st.session_state.model_name}`")
st.sidebar.write(f"**Threshold:** `{st.session_state.threshold}`")
st.sidebar.caption("Class = 1 if prob ≥ threshold, else 0.")

st.sidebar.divider()
st.sidebar.subheader("Artifacts (GitHub)")
st.sidebar.markdown(f"- **pipeline.pkl**: `{PIPELINE_URL}`")
st.sidebar.markdown(f"- **metadata.json**: `{METADATA_URL}`")

# -----------------------------
# Header
# -----------------------------
st.title("Real Time Scoring App (powered by SAS)")
st.write("Score individual cases or batches in real time using the latest model stored in GitHub.")

tabs = st.tabs(["Single Case Form", "Batch CSV Upload", "Chat Assistant"])

# =========================================================
# Tab 1: Single Case Form
# =========================================================
with tabs[0]:
    st.subheader("Single Case Form")
    st.caption("Fill the inputs below and click **Score**. Inputs are generated from the model metadata.")

    # dynamic form from metadata inputs
    form_vals = {}
    with st.form("single_case_form"):
        cols = st.columns(2)
        for i, spec in enumerate(st.session_state.inputs_spec):
            nm = spec.get("name", f"col_{i}")
            typ = (spec.get("type") or "").lower()
            required = bool(spec.get("required", False))

            col = cols[i % 2]
            if typ == "number":
                form_vals[nm] = col.number_input(
                    label=nm, value=0.0, step=0.1, format="%.4f"
                )
            else:
                form_vals[nm] = col.text_input(label=nm, value="")
        submit = st.form_submit_button("Score")

    if submit:
        try:
            X = pd.DataFrame([form_vals])
            with st.spinner("Scoring..."):
                prob, lab = predict_df(model, X)
            st.success("Done")
            st.metric("Predicted Probability (class=1)", f"{prob.iloc[0]:.4f}")
            st.metric("Predicted Label", int(lab.iloc[0]))
            st.info(f"Threshold used: **{st.session_state.threshold}**")
            with st.expander("Scored row (features + outputs)"):
                out = X.copy()
                out["prob_1"] = prob.values
                out["label"] = lab.values
                st.dataframe(out, use_container_width=True)
        except Exception as e:
            st.error(f"Error scoring the record:\n{e}")

# =========================================================
# Tab 2: Batch CSV Upload
# =========================================================
with tabs[1]:
    st.subheader("Batch CSV Upload")
    st.caption("Upload a CSV with the expected columns (as per the model metadata).")
    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            st.write("**Preview:**")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("Score CSV"):
                with st.spinner("Scoring batch..."):
                    prob, lab = predict_df(model, df)
                    out = df.copy()
                    out["prob_1"] = prob.values
                    out["label"] = lab.values
                st.success("Batch scored.")
                st.write("**Results (first 100 rows):**")
                st.dataframe(out.head(100), use_container_width=True)

                csv_buf = StringIO()
                out.to_csv(csv_buf, index=False)
                st.download_button(
                    "Download scored CSV",
                    data=csv_buf.getvalue(),
                    file_name="scored_output.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Failed to read/score CSV:\n{e}")

# =========================================================
# Tab 3: Chat Assistant (guided, no external LLM required)
# =========================================================
with tabs[2]:
    st.subheader("Chat Assistant")
    st.caption("A lightweight guided assistant to collect inputs and return a score.")

    # Session state for chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_step" not in st.session_state:
        st.session_state.chat_step = 0
    if "chat_inputs" not in st.session_state:
        st.session_state.chat_inputs = build_empty_row(st.session_state.inputs_spec)

    def push_msg(role: str, content: str):
        st.session_state.chat_history.append({"role": role, "content": content})

    # initial system message
    if not st.session_state.chat_history:
        push_msg("assistant", "Hi! I can help you score a case. Say 'start' to begin, or paste values like 'Age=45, Income=50000'.")

    # render history
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_msg = st.chat_input("Type here...")
    if user_msg:
        # show user msg
        with st.chat_message("user"):
            st.write(user_msg)
        push_msg("user", user_msg)

        # simple state machine
        try:
            if user_msg.strip().lower() in ["start", "begin", "go"]:
                st.session_state.chat_step = 0
                st.session_state.chat_inputs = build_empty_row(st.session_state.inputs_spec)
                if st.session_state.inputs_spec:
                    q = f"Please provide **{st.session_state.inputs_spec[0]['name']}**:"
                else:
                    q = "No input schema found in metadata. Please use the form or CSV."
                push_msg("assistant", q)

            elif "=" in user_msg:
                # quick parse: "Age=45, Income=50000"
                pairs = [p.strip() for p in user_msg.split(",")]
                for p in pairs:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        # coerce to number if input schema says number
                        spec_map = {x["name"]: (x.get("type") or "").lower() for x in st.session_state.inputs_spec}
                        if spec_map.get(k, "") == "number":
                            try:
                                st.session_state.chat_inputs[k] = float(v)
                            except Exception:
                                pass
                        else:
                            st.session_state.chat_inputs[k] = v
                # score immediately
                X = pd.DataFrame([st.session_state.chat_inputs])
                prob, lab = predict_df(model, X)
                msg = f"**Probability**: {prob.iloc[0]:.4f} | **Label**: {int(lab.iloc[0])} (thr={st.session_state.threshold})"
                push_msg("assistant", msg)

            else:
                # guided Q&A
                specs = st.session_state.inputs_spec
                idx = st.session_state.chat_step
                if 0 <= idx < len(specs):
                    nm = specs[idx]["name"]
                    typ = (specs[idx].get("type") or "").lower()
                    val = user_msg.strip()
                    if typ == "number":
                        try:
                            val = float(val)
                        except Exception:
                            push_msg("assistant", f"Please enter a numeric value for **{nm}**.")
                            st.stop()
                    st.session_state.chat_inputs[nm] = val
                    st.session_state.chat_step += 1
                    # next question or score
                    if st.session_state.chat_step < len(specs):
                        nm2 = specs[st.session_state.chat_step]["name"]
                        push_msg("assistant", f"Please provide **{nm2}**:")
                    else:
                        X = pd.DataFrame([st.session_state.chat_inputs])
                        prob, lab = predict_df(model, X)
                        msg = f"**Probability**: {prob.iloc[0]:.4f} | **Label**: {int(lab.iloc[0])} (thr={st.session_state.threshold})"
                        push_msg("assistant", msg)
                        push_msg("assistant", "Type 'start' to score another case, or provide key=value pairs.")
        except Exception as e:
            push_msg("assistant", f"Sorry, I hit an error: {e}")
