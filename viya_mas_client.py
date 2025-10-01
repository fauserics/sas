# viya_mas_client.py
# Minimal, robust MAS client for SAS Viya scoring from Streamlit/GitHub.
# Works with: BEARER_TOKEN / SAS_SERVICES_TOKEN, or password grant via SASLogon.
# Also lets you pull the module input schema and save it to score/inputVar.json.

import os
import json
import base64
import requests
from typing import Dict, List, Optional, Tuple

# --------- ENV VARS (set in Streamlit Secrets or environment) ----------
# VIYA_URL           : https://your-viya.domain
# MAS_MODULE_ID      : module id/uuid (from publish step)
# BEARER_TOKEN       : optional pre-issued OAuth/JWT
# SAS_SERVICES_TOKEN : optional service-to-service token (same as bearer)
# VIYA_USER          : optional, username (password grant)
# VIYA_PASSWORD      : optional, password (password grant)
# SAS_CLIENT_ID      : optional, oauth client id (default: viya_client)
# SAS_CLIENT_SECRET  : optional, oauth client secret
# OAUTH_TOKEN_URL    : optional, defaults to {VIYA_URL}/SASLogon/oauth/token
# VIYA_TLS_VERIFY    : "0" or "1" (default 1). If "0", skip TLS verify (dev only!)
# VIYA_CA_BUNDLE     : optional path to CA bundle file to verify TLS

def _bool_env(name: str, default: bool = True) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip() not in ("0", "false", "False", "NO", "no")

def _session() -> requests.Session:
    s = requests.Session()
    verify = _bool_env("VIYA_TLS_VERIFY", True)
    ca = os.getenv("VIYA_CA_BUNDLE", None)
    if not verify:
        s.verify = False
    elif ca:
        s.verify = ca
    # reasonable timeouts
    s.timeout = (10, 30)  # (connect, read)
    return s

def _get_base() -> str:
    base = (os.getenv("VIYA_URL") or "").rstrip("/")
    if not base:
        raise RuntimeError("VIYA_URL is not set.")
    return base

def _bearer_from_env() -> Optional[str]:
    tok = os.getenv("BEARER_TOKEN") or os.getenv("SAS_SERVICES_TOKEN")
    if tok:
        return tok.strip()
    return None

def _oauth_token_url() -> str:
    base = _get_base()
    return os.getenv("OAUTH_TOKEN_URL", f"{base}/SASLogon/oauth/token")

def _get_client_auth() -> Tuple[str, str]:
    cid = os.getenv("SAS_CLIENT_ID", "viya_client")
    csc = os.getenv("SAS_CLIENT_SECRET", "")
    return cid, csc

def _get_password_creds() -> Optional[Tuple[str, str]]:
    u, p = os.getenv("VIYA_USER"), os.getenv("VIYA_PASSWORD")
    if u and p:
        return u, p
    return None

def _fetch_token_with_password_grant() -> Optional[str]:
    creds = _get_password_creds()
    if not creds:
        return None
    cid, csc = _get_client_auth()
    token_url = _oauth_token_url()
    sess = _session()
    data = {
        "grant_type": "password",
        "username": creds[0],
        "password": creds[1],
        "scope": "openid"
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    # Basic auth with client_id:client_secret
    auth = None
    if cid:
        # If client secret is empty, most Viya setups still allow well-known public clients.
        auth = (cid, csc)
    r = sess.post(token_url, data=data, headers=headers, auth=auth)
    if r.status_code >= 400:
        raise RuntimeError(f"Token error ({r.status_code}): {r.text}")
    tok = r.json().get("access_token")
    if not tok:
        raise RuntimeError("Token response missing access_token.")
    return tok

def _get_token() -> str:
    # 1) Try env-provided token
    t = _bearer_from_env()
    if t:
        return t
    # 2) Try password grant
    t = _fetch_token_with_password_grant()
    if t:
        return t
    raise RuntimeError("No token available. Set BEARER_TOKEN or configure VIYA_USER/VIYA_PASSWORD (+ SAS_CLIENT_ID/SECRET).")

def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {_get_token()}"}

def _module_id() -> str:
    mid = os.getenv("MAS_MODULE_ID", "").strip()
    if not mid:
        raise RuntimeError("MAS_MODULE_ID is not set.")
    return mid

# ---------------------- PUBLIC API ---------------------------------

def get_module_info() -> Dict:
    """Fetch MAS module metadata (includes inputs/outputs schema)."""
    base = _get_base()
    mid = _module_id()
    url = f"{base}/microanalyticScore/modules/{mid}"
    sess = _session()
    r = sess.get(url, headers=_auth_headers())
    if r.status_code >= 400:
        raise RuntimeError(f"get_module_info error ({r.status_code}): {r.text}")
    return r.json()

def get_module_inputs() -> List[Dict]:
    """Return a simplified list: [{'name':..., 'type':...}, ...]"""
    info = get_module_info()
    inputs = info.get("inputs") or []
    out = []
    for it in inputs:
        out.append({"name": it.get("name"), "type": it.get("type")})
    return out

def build_example_payload(row: Dict) -> Dict:
    """Map a python dict row -> MAS payload {'inputs':[{'name':..,'value':..},...] }"""
    # MAS is strict with names; use as-is. You may add mapping here if needed.
    payload = {"inputs": []}
    for k, v in row.items():
        payload["inputs"].append({"name": k, "value": v})
    return payload

def score_row_via_rest(row: Dict) -> Dict:
    """
    POST /microanalyticScore/modules/{id}/steps/score
    Returns raw json dict (MAS format). The app parses it with parse_mas_outputs().
    """
    base = _get_base()
    mid = _module_id()
    url = f"{base}/microanalyticScore/modules/{mid}/steps/score"
    sess = _session()
    payload = build_example_payload(row)
    r = sess.post(url, json=payload, headers=_auth_headers())
    if r.status_code >= 400:
        raise RuntimeError(f"score_row_via_rest error ({r.status_code}): {r.text}")
    return r.json()

def sync_input_schema_to_file(path: str = "score/inputVar.json") -> List[str]:
    """
    Pull input schema from MAS and write to score/inputVar.json.
    Returns the list of variable names.
    """
    vars = get_module_inputs()
    names = [v["name"] for v in vars if v.get("name")]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"inputVariables": [{"name": n} for n in names]}, f, indent=2)
    return names

def ping() -> bool:
    """Simple check: can we read module info?"""
    try:
        _ = get_module_info()
        return True
    except Exception:
        return False
