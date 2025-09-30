# sas_code_translator.py
# Traduce score SAS (DATA step/DS2 con asignaciones + IF/ELSE) a Python.
# Ejecuta el cuerpo traducido dentro de un 'env' (exec), y garantiza
# que el dict devuelto tenga EM_EVENTPROBABILITY (buscando P_* si falta).

import re, math, types
import pandas as pd
from typing import Callable, List, Tuple

# ===== helpers de runtime (disponibles para el score traducido) =====
def MISSING(x):
    return x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and x == "")

def MEAN(*args):
    vals = [float(v) for v in args if not MISSING(v)]
    return sum(vals)/len(vals) if vals else float("nan")

def SUM(*args):
    return sum(float(v) for v in args if not MISSING(v))

def NMISS(*args):
    return sum(1 for v in args if MISSING(v))

def COALESCE(*args):
    for v in args:
        if not MISSING(v):
            return v
    return None

# ===== mapeos de operadores/funciones SAS -> Python =====
_OP_MAP = {
    r"\bEQ\b": "==", r"\bNE\b": "!=", r"\bGE\b": ">=", r"\bLE\b": "<=",
    r"\bGT\b": ">",  r"\bLT\b": "<",  r"\bAND\b": "and", r"\bOR\b": "or", r"\bNOT\b": "not"
}
_FUNC_MAP = {
    r"\bLOG\(": "math.log(", r"\bEXP\(": "math.exp(", r"\bABS\(": "abs(",
    r"\bSQRT\(": "math.sqrt(", r"\bMAX\(": "max(", r"\bMIN\(": "min(",
    r"\bMEAN\(": "MEAN(", r"\bSUM\(": "SUM(", r"\bNMISS\(": "NMISS(", r"\bCOALESCE\(": "COALESCE("
}

# ===== util de parsing =====
def _strip_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)          # /* ... */
    s = re.sub(r"^\s*\*.*?;\s*$", "", s, flags=re.M)     # * ... ;
    return s

def _split_statements(sas_code: str) -> List[str]:
    # divide por ';' respetando comillas
    out, buf, q = [], [], None
    for ch in sas_code:
        if q:
            buf.append(ch)
            if ch == q: q = None
        else:
            if ch in ("'", '"'):
                q = ch; buf.append(ch)
            elif ch == ';':
                out.append(''.join(buf).strip()); buf = []
            else:
                buf.append(ch)
    if buf: out.append(''.join(buf).strip())
    return [x for x in out if x]

def _clean_stmt(stmt: str) -> str:
    # ignorar LABEL/LENGTH/FORMAT/DROP/KEEP/DECLARE/RETAIN/ARRAY/RETURN y cabeceras DS2
    if re.match(r"^\s*(label|length|format|drop|keep|dcl|declare|retain|array|return)\b", stmt, flags=re.I):
        return ""
    if re.match(r"^\s*(package|method|end|output|run)\b", stmt, flags=re.I):
        return ""  # estructura DS2/packaging
    # concatenación y missing
    stmt = stmt.replace("||", "+")
    stmt = re.sub(r"=\s*\.", "= None", stmt)             # asignación a missing
    # operadores/funciones
    for k,v in _OP_MAP.items():   stmt = re.sub(k, v, stmt, flags=re.I)
    for k,v in _FUNC_MAP.items(): stmt = re.sub(k, v, stmt, flags=re.I)
    # En IF: '=' -> '==' (sin tocar asignaciones)
    if re.match(r"^\s*if\b", stmt, flags=re.I):
        stmt = re.sub(r"(?<![<>=!])=(?![=])", "==", stmt)
    return stmt.strip()

def translate_sas_to_python_body(sas_code: str) -> Tuple[str, List[str]]:
    """
    Devuelve (cuerpo_python, discovered_inputs).
    El cuerpo es una serie de sentencias Python (sin 'def') listas para exec().
    """
    code = _strip_comments(sas_code)
    stmts = _split_statements(code)

    py_lines = []
    indent = 0
    inputs = set()
    created = set()

    re_if   = re.compile(r"^\s*if\b(.*)\bthen\b\s*(do)?\s*$", flags=re.I)
    re_else = re.compile(r"^\s*else\b\s*(do)?\s*$", flags=re.I)
    re_end  = re.compile(r"^\s*end\s*$", flags=re.I)
    re_asg  = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+)$")

    for raw in stmts:
        st = _clean_stmt(raw)
        if not st:
            continue
        if re_end.match(st):
            indent = max(0, indent-1); continue
        m_if = re_if.match(st)
        if m_if:
            cond = m_if.group(1).strip()
            py_lines.append("    "*indent + f"if {cond}:")
            indent += 1
            continue
        if re_else.match(st):
            indent = max(0, indent-1)
            py_lines.append("    "*indent + "else:")
            indent += 1
            continue
        m_as = re_asg.match(st)
        if m_as:
            var, expr = m_as.group(1), m_as.group(2)
            created.add(var.upper())
            py_lines.append("    "*indent + f"{var} = {expr}")
            # inputs heurísticos
            for tok in re.findall(r"[A-Za-z_]\w*", expr):
                if tok.upper() not in {
                    "IF","ELSE","AND","OR","NOT","MISSING","MEAN","SUM","NMISS","COALESCE",
                    "LOG","EXP","ABS","SQRT","MAX","MIN","MATH","NONE","NAN","TRUE","FALSE"
                } and tok.upper() not in created:
                    inputs.add(tok)
            continue
        # otras sentencias: omitir

    body = "\n".join(py_lines) if py_lines else "pass"
    return body, sorted(inputs)

# ===== compilación a función Python =====
def _pick_prob_from_env(env: dict) -> float | None:
    # 1) EM_EVENTPROBABILITY directo
    if "EM_EVENTPROBABILITY" in env and env["EM_EVENTPROBABILITY"] is not None:
        return float(env["EM_EVENTPROBABILITY"])
    # 2) cualquier P_*, con preferencia por sufijo '1' o que contenga 'BAD1'/'EVENT'
    p_keys = [k for k in env.keys() if k.upper().startswith("P_") and env[k] is not None]
    if not p_keys:
        return None
    # preferidos
    def key_score(k):
        ku = k.upper()
        score = 0
        if ku.endswith("1"): score += 3
        if "BAD1" in ku or "EVENT" in ku: score += 2
        if ku in ("P_BAD", "P_TARGET1"): score += 1
        return score
    p_keys.sort(key=lambda k: (key_score(k), k), reverse=True)
    try:
        return float(env[p_keys[0]])
    except Exception:
        return None

def compile_sas_score(sas_code: str, func_name: str = "sas_score", expected_inputs: list[str] | None = None
                      ) -> Tuple[Callable, str, List[str]]:
    """
    Compila el score SAS a una función Python invocable: score_fn(**row_dict) -> dict
    Devuelve (score_fn, python_code_for_display, expected_inputs_final)
    """
    body, discovered_inputs = translate_sas_to_python_body(sas_code)
    inputs = expected_inputs or discovered_inputs or []

    # función "cerrada" que ejecuta el cuerpo en un env (locals) controlado
    def _score_fn(**row):
        env = {}
        # helpers y libs disponibles para el body:
        env.update({
            "math": math,
            "MISSING": MISSING, "MEAN": MEAN, "SUM": SUM, "NMISS": NMISS, "COALESCE": COALESCE
        })
        # inputs conocidos (en minúscula/mayúscula tal cual vienen)
        for name in inputs:
            env[name] = row.get(name, None)
        # además, exponer todo lo que venga (por si el score usa algún alias no listado)
        for k, v in row.items():
            if k not in env:
                env[k] = v
        # ejecutar el cuerpo traducido
        exec(body, {}, env)

        # armar salida robusta
        out = {}
        p = _pick_prob_from_env(env)
        if p is not None:
            out["EM_EVENTPROBABILITY"] = p
        # clasificación si está
        for key in ("EM_CLASSIFICATION", "I_BAD", "I_TARGET", "LABEL"):
            if key in env and env[key] is not None:
                out["EM_CLASSIFICATION"] = env[key]
                break
        return out

    # código para mostrar en la UI (legible)
    display_code = (
        "def " + func_name + "(**row):\n"
        "    # inputs available: " + ", ".join(inputs) + "\n"
        "    # --- translated SAS body ---\n" +
        "\n".join(["    " + ln for ln in body.splitlines()]) + "\n" +
        "    # --- end translated body ---\n"
        "    # (runtime builds EM_EVENTPROBABILITY from P_* if needed)\n"
    )

    return _score_fn, display_code, inputs

# ===== helper para DataFrame =====
def score_dataframe(score_fn: Callable, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        out = score_fn(**r.to_dict()) or {}
        # prob
        p = out.get("EM_EVENTPROBABILITY", None)
        if p is None:
            # sin prob: es un error lógico del score traducido -> mensaje claro
            raise ValueError("Translated score returned no probability. "
                             "Make sure the SAS code assigns P_* or EM_EVENTPROBABILITY.")
        p = float(p)
        # label
        lab = out.get("EM_CLASSIFICATION", None)
        if lab is None:
            lab = 1 if p >= threshold else 0
        else:
            try:
                lab = int(lab)
            except Exception:
                lab = 1 if p >= threshold else 0
        rec = r.to_dict()
        rec["prob_BAD"] = p
        rec["label_BAD"] = int(lab)
        rows.append(rec)
    return pd.DataFrame(rows)
