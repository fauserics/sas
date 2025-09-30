# viya_mas_client.py
# Minimal MAS REST client.

import os, base64, requests

def _bearer():
    tok = os.getenv("BEARER_TOKEN") or os.getenv("SAS_SERVICES_TOKEN")
    if tok: return tok
    # Password grant fallback (if configured)
    cid, csec = os.getenv("VIYA_CLIENT_ID"), os.getenv("VIYA_CLIENT_SECRET")
    user, pwd = os.getenv("VIYA_USER"), os.getenv("VIYA_PASSWORD")
    if not all([cid, csec, user, pwd]):
        raise RuntimeError("No token or OAuth credentials found (set BEARER_TOKEN or SAS_SERVICES_TOKEN, or VIYA_CLIENT_ID/SECRET + VIYA_USER/PASSWORD).")
    auth = base64.b64encode(f"{cid}:{csec}".encode()).decode()
    r = requests.post(
        f"{os.getenv('VIYA_URL')}/SASLogon/oauth/token",
        headers={"Authorization": f"Basic {auth}","Content-Type":"application/x-www-form-urlencoded"},
        data={"grant_type":"password","username":user,"password":pwd}, timeout=30
    )
    r.raise_for_status()
    return r.json()["access_token"]

def score_row_via_rest(row: dict):
    base = os.getenv("VIYA_URL")
    mod  = os.getenv("MAS_MODULE_ID")
    if not base or not mod:
        raise RuntimeError("VIYA_URL and MAS_MODULE_ID are required.")
    # Typical MAS step: 'score'. Adjust if your deployment uses 'steps/execute'.
    url = f"{base}/microanalyticScore/modules/{mod}/steps/score"
    hdr = {"Authorization": f"Bearer {_bearer()}", "Content-Type": "application/json"}
    body = {"inputs": [{"name":k, "value":v} for k,v in row.items()]}
    r = requests.post(url, json=body, headers=hdr, timeout=30)
    r.raise_for_status()
    return r.json()
