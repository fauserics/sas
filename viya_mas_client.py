# viya_mas_client.py
import os, time, json, requests
from typing import Dict, Any, Optional

def _get_env(name: str, required: bool = False) -> Optional[str]:
    v = os.getenv(name)
    if required and not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def _build_session(retries: int = 2, backoff: float = 0.5) -> requests.Session:
    s = requests.Session()
    s.retries = retries  # marker propio (no urllib3 Retry para mantener deps simples)
    s.backoff = backoff
    return s

def _post_json_with_simple_retry(
    sess: requests.Session,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
    verify: Any
) -> requests.Response:
    last_exc = None
    attempts = getattr(sess, "retries", 2) + 1
    backoff = getattr(sess, "backoff", 0.5)
    for i in range(attempts):
        try:
            r = sess.post(url, json=payload, headers=headers, timeout=timeout, verify=verify)
            # 401/403/5xx → igualmente hacemos raise_for_status para que la app lo informe
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            if i < attempts - 1:
                time.sleep(backoff * (2 ** i))
            else:
                raise
    # nunca llega
    raise last_exc

def score_row_via_rest(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Env vars requeridas:
      - VIYA_URL           (p.ej. https://disco-a20237-rg.gelenable.sas.com)
      - MAS_MODULE_ID      (id del módulo MAS del modelo GB)
      - BEARER_TOKEN  ó SAS_SERVICES_TOKEN  (access_token válido de SAS Logon)
    Opcionales:
      - VIYA_TIMEOUT       (segundos; default 30)
      - VIYA_CA_BUNDLE     (ruta a PEM corporativo; si no está, usa verify=True)
    """
    base = _get_env("VIYA_URL", required=True).rstrip("/")
    module_id = _get_env("MAS_MODULE_ID", required=True)
    token = os.getenv("BEARER_TOKEN") or os.getenv("SAS_SERVICES_TOKEN")
    if not token:
        raise RuntimeError("Missing BEARER_TOKEN (o SAS_SERVICES_TOKEN).")

    url = f"{base}/microanalyticScore/modules/{module_id}/steps/score"
    timeout = float(os.getenv("VIYA_TIMEOUT", "30"))
    verify = os.getenv("VIYA_CA_BUNDLE") or True

    # MAS espera: {"inputs":[{"name":"VAR","value":...}, ...]}
    payload = {"inputs": [{"name": k, "value": v} for k, v in (row or {}).items()]}

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    sess = _build_session(retries=int(os.getenv("VIYA_RETRIES", "2")), backoff=0.5)
    resp = _post_json_with_simple_retry(sess, url, payload, headers, timeout, verify)
    # devuelve el JSON crudo; app.py ya llama a parse_mas_outputs(...)
    return resp.json()

def get_module_signature() -> Dict[str, Any]:
    """Utilidad opcional para depurar: trae metadata del módulo (inputs/outputs)."""
    base = _get_env("VIYA_URL", required=True).rstrip("/")
    module_id = _get_env("MAS_MODULE_ID", required=True)
    token = os.getenv("BEARER_TOKEN") or os.getenv("SAS_SERVICES_TOKEN")
    if not token:
        raise RuntimeError("Missing BEARER_TOKEN (o SAS_SERVICES_TOKEN).")
    url = f"{base}/microanalyticScore/modules/{module_id}"
    verify = os.getenv("VIYA_CA_BUNDLE") or True
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=15, verify=verify)
    r.raise_for_status()
    return r.json()
