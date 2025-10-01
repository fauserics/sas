# viya_mas_client.py
# Cliente MAS para SAS Viya con bypass de proxy opcional, timeouts explícitos y diagnóstico.

import os, json, socket, requests
from typing import Dict, List, Optional, Tuple

# ----------------- Helpers de configuración -----------------

def _get_base() -> str:
    base = (os.getenv("VIYA_URL") or "").rstrip("/")
    if not base:
        raise RuntimeError("VIYA_URL is not set.")
    return base

def _module_id() -> str:
    mid = (os.getenv("MAS_MODULE_ID") or "").strip()
    if not mid:
        raise RuntimeError("MAS_MODULE_ID is not set.")
    return mid

def _bool_env(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() not in ("0","false","no")

def _verify_arg():
    # TLS verify según secrets
    if not _bool_env("VIYA_TLS_VERIFY", True):
        return False
    cab = os.getenv("VIYA_CA_BUNDLE")
    return cab if cab else True

def _timeouts() -> Tuple[float, float]:
    # Timeouts explícitos (connect, read). Defaults seguros.
    def _f(k, d):
        try:
            val = os.getenv(k)
            return float(val) if val not in (None, "", "None") else float(d)
        except Exception:
            return float(d)
    return (_f("VIYA_TIMEOUT_CONNECT", 30.0), _f("VIYA_TIMEOUT_READ", 120.0))

def _token_from_env() -> Optional[str]:
    tok = (os.getenv("BEARER_TOKEN") or os.getenv("SAS_SERVICES_TOKEN") or "").strip()
    if not tok:
        return None
    return tok.replace("\n","").replace("\r","").strip()

def _oauth_token_url() -> str:
    return os.getenv("OAUTH_TOKEN_URL", f"{_get_base()}/SASLogon/oauth/token")

def _client_auth() -> Tuple[str, str]:
    return os.getenv("SAS_CLIENT_ID", "viya_client"), os.getenv("SAS_CLIENT_SECRET", "")

def _password_creds() -> Optional[Tuple[str, str]]:
    u, p = os.getenv("VIYA_USER"), os.getenv("VIYA_PASSWORD")
    if u and p: return u, p
    return None

# ----------------- Sesión HTTP -----------------

def _session(no_proxy: bool = False) -> requests.Session:
    s = requests.Session()
    # Si querés ignorar completamente variables de proxy del entorno:
    force = os.getenv("VIYA_FORCE_NO_PROXY", "").strip().lower() in ("1","true","yes")
    if no_proxy or force:
        s.trust_env = False
        s.proxies.clear()
    else:
        s.trust_env = True  # respeta HTTPS_PROXY/HTTP_PROXY/NO_PROXY
        p = os.getenv("PROXY_URL")
        if p:
            s.proxies.update({"http": p, "https": p})
    return s

# ----------------- OAuth (si no hay token en secrets) -----------------

def _fetch_token_with_password_grant() -> str:
    creds = _password_creds()
    if not creds:
        raise RuntimeError("No token and no VIYA_USER/VIYA_PASSWORD provided.")
    cid, csc = _client_auth()
    r = _session().post(
        _oauth_token_url(),
        data={"grant_type":"password","username":creds[0],"password":creds[1],"scope":"openid"},
        auth=(cid, csc),
        headers={"Content-Type":"application/x-www-form-urlencoded"},
        verify=_verify_arg(), timeout=_timeouts()
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Token error ({r.status_code}): {r.text}")
    tok = r.json().get("access_token")
    if not tok:
        raise RuntimeError("Token response missing access_token.")
    return tok

def _get_token() -> str:
    tok = _token_from_env()
    if tok: return tok
    return _fetch_token_with_password_grant()

def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {_get_token()}"}

# ----------------- API pública -----------------

def ping() -> bool:
    try:
        _ = list_modules(limit=1)
        return True
    except Exception:
        return False

def list_modules(limit: int = 200) -> List[Dict]:
    url = f"{_get_base()}/microanalyticScore/modules?limit={int(limit)}"
    r = _session().get(url, headers=_auth_headers(), verify=_verify_arg(), timeout=_timeouts())
    if r.status_code >= 400:
        raise RuntimeError(f"list_modules error ({r.status_code}): {r.text}")
    data = r.json()
    items = data.get("items") if isinstance(data, dict) else data
    return items or []

def get_module_info(module_id: Optional[str] = None) -> Dict:
    mid = module_id or _module_id()
    url = f"{_get_base()}/microanalyticScore/modules/{mid}"
    r = _session().get(url, headers=_auth_headers(), verify=_verify_arg(), timeout=_timeouts())
    if r.status_code >= 400:
        raise RuntimeError(f"get_module_info error ({r.status_code}): {r.text}")
    return r.json()

def get_module_inputs(module_id: Optional[str] = None) -> List[Dict]:
    info = get_module_info(module_id=module_id)
    inputs = info.get("inputs") or []
    return [{"name": it.get("name"), "type": it.get("type")} for it in inputs]

def build_example_payload(row: Dict) -> Dict:
    return {"inputs": [{"name": k, "value": v} for k, v in row.items()]}

def score_row_via_rest(row: Dict, module_id: Optional[str] = None) -> Dict:
    mid = module_id or _module_id()
    url = f"{_get_base()}/microanalyticScore/modules/{mid}/steps/score"
    # *** Importante: timeout explícito y sesión sin proxy si así se pide ***
    sess = _session()
    r = sess.post(url, json=build_example_payload(row),
                  headers=_auth_headers(), verify=_verify_arg(),
                  timeout=_timeouts())
    if r.status_code >= 400:
        raise RuntimeError(f"score_row_via_rest error ({r.status_code}): {r.text}")
    return r.json()

# Variante "directa" que ignora proxy (para diagnóstico puntual)
def score_row_via_rest_noproxy(row: Dict, module_id: Optional[str] = None) -> Dict:
    mid = module_id or _module_id()
    url = f"{_get_base()}/microanalyticScore/modules/{mid}/steps/score"
    r = _session(no_proxy=True).post(
        url, json=build_example_payload(row),
        headers=_auth_headers(), verify=_verify_arg(),
        timeout=_timeouts()
    )
    if r.status_code >= 400:
        raise RuntimeError(f"score_row_via_rest_noproxy error ({r.status_code}): {r.text}")
    return r.json()

def sync_input_schema_to_file(path: str = "score/inputVar.json", module_id: Optional[str] = None) -> List[str]:
    vars = get_module_inputs(module_id=module_id)
    names = [v["name"] for v in vars if v.get("name")]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"inputVariables": [{"name": n} for n in names]}, f, indent=2)
    return names

# ----------------- Diagnóstico -----------------

def connectivity_check() -> Dict:
    info = {
        "base": _get_base(),
        "timeouts": _timeouts(),
        "env": {
            "HTTPS_PROXY": os.getenv("HTTPS_PROXY"),
            "HTTP_PROXY": os.getenv("HTTP_PROXY"),
            "NO_PROXY": os.getenv("NO_PROXY"),
            "VIYA_FORCE_NO_PROXY": os.getenv("VIYA_FORCE_NO_PROXY"),
            "VIYA_TLS_VERIFY": os.getenv("VIYA_TLS_VERIFY"),
        }
    }
    try:
        host = info["base"].split("://",1)[-1].split("/",1)[0]
        info["dns"] = {"host": host, "ip": socket.gethostbyname(host)}
    except Exception as e:
        info["dns_error"] = repr(e)

    # intentamos listar módulos con y sin proxy
    try:
        url = f"{info['base']}/microanalyticScore/modules?limit=1"
        r = _session().get(url, headers=_auth_headers(), verify=_verify_arg(), timeout=_timeouts())
        info["list_with_proxy"] = {"status": r.status_code, "ok": (r.status_code==200)}
        if r.status_code != 200:
            info["list_with_proxy"]["body"] = r.text[:400]
    except Exception as e:
        info["list_with_proxy_error"] = repr(e)

    try:
        url = f"{info['base']}/microanalyticScore/modules?limit=1"
        r = _session(no_proxy=True).get(url, headers=_auth_headers(), verify=_verify_arg(), timeout=_timeouts())
        info["list_noproxy"] = {"status": r.status_code, "ok": (r.status_code==200)}
        if r.status_code != 200:
            info["list_noproxy"]["body"] = r.text[:400]
    except Exception as e:
        info["list_noproxy_error"] = repr(e)

    return info
