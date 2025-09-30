# viya_mas_client.py
import os, base64, requests

def _bearer():
    tok = os.getenv("BEARER_TOKEN") or os.getenv("SAS_SERVICES_TOKEN")
    if tok: return tok
    # Password grant (si decidís usar user/pass + client)
    cid, csec = os.getenv("VIYA_CLIENT_ID"), os.getenv("VIYA_CLIENT_SECRET")
    user, pwd = os.getenv("VIYA_USER"), os.getenv("VIYA_PASSWORD")
    auth = base64.b64encode(f"{cid}:{csec}".encode()).decode()
    r = requests.post(
        f"{os.getenv('VIYA_URL')}/SASLogon/oauth/token",
        headers={"Authorization": f"Basic {auth}","Content-Type":"application/x-www-form-urlencoded"},
        data={"grant_type":"password","username":user,"password":pwd}, timeout=30
    )
    r.raise_for_status()
    return r.json()["access_token"]

def score_row_via_rest(row: dict):
    url = f"{os.getenv('VIYA_URL')}/microanalyticScore/modules/{os.getenv('MAS_MODULE_ID')}/steps/score"
    hdr = {"Authorization": f"Bearer {_bearer()}", "Content-Type": "application/json"}
    body = {"inputs": [{"name":k, "value":v} for k,v in row.items()]}
    r = requests.post(url, json=body, headers=hdr, timeout=30)
    r.raise_for_status()
    return r.json()  # parseá outputs según tu módulo
