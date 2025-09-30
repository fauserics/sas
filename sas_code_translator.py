# sas_code_translator.py
import re, math, types
import pandas as pd
from typing import Callable, List, Tuple

# ---------- helpers disponibles en el runtime traducido ----------
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

# ---------- traducción línea a línea (DATA step score) ----------
_OP_MAP = {
    r"\bEQ\b": "==", r"\bNE\b": "!=", r"\bGE\b": ">=", r"\bLE\b": "<=",
    r"\bGT\b": ">",  r"\bLT\b": "<",  r"\bAND\b": "and", r"\bOR\b": "or", r"\bNOT\b": "not"
}
_FUNC_MAP = {
    r"\bLOG\(": "math.log(", r"\bEXP\(": "math.exp(", r"\bABS\(": "abs(",
    r"\bSQRT\(": "math.sqrt(", r"\bMAX\(": "max(", r"\bMIN\(": "min(",
    r"\bMEAN\(": "MEAN(", r"\bSUM\(": "SUM(", r"\bNMISS\(": "NMISS(", r"\bCOALESCE\(": "COALESCE("
}

def _strip_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)          # /* ... */
    s = re.sub(r"^\s*\*.*?;\s*$", "", s, flags=re.M)     # * ... ;
    return s

def _clean_stmt(stmt: str) -> str:
    # normaliza espacios, quita LABEL/LENGTH/FORMAT/DROP/KEEP
    if re.match(r"^\s*(label|length|format|drop|keep)\b", stmt, flags=re.I):
        return ""
    # concatenación y missings
    stmt = stmt.replace("||", "+")
    stmt = re.sub(r"=\s*\.", "= None", stmt)             # asignación a missing
    # funciones y operadores
    for k,v in _OP_MAP.items():   stmt = re.sub(k, v, stmt, flags=re.I)
    for k,v in _FUNC_MAP.items(): stmt = re.sub(k, v, stmt, flags=re.I)
    # MISSING(x) -> MISSING(x)
    # comparaciones dentro de IF: = -> == (pero no romper asignaciones)
    if re.match(r"^\s*if\b", stmt, flags=re.I):
        # proteger >=, <=, !=, == ya presentes
        stmt = re.sub(r"(?<![<>=!])=(?![=])", "==", stmt)
    return stmt.strip()

def _split_statements(sas_code: str) -> List[str]:
    # divide por ';' respetando comillas simples/dobles
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

def translate_sas_to_python(sas_code: str, func_name="sas_score") -> Tuple[str, List[str]]:
    """
    Devuelve (codigo_python, expected_inputs).
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
    re_assign = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+)$")

    for raw in stmts:
        st = _clean_stmt(raw)
        if not st: continue
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
        m_as = re_assign.match(st)
        if m_as:
            var, expr = m_as.group(1), m_as.group(2)
            created.add(var.upper())
            py_lines.append("    "*indent + f"{var} = {expr}")
            # recolectar posibles inputs simples (heurística)
            for tok in re.findall(r"[A-Za-z_]\w*", expr):
                if tok.upper() not in {"IF","ELSE","AND","OR","NOT","MISSING","MEAN","SUM","NMISS","COALESCE",
                                       "LOG","EXP","ABS","SQRT","MAX","MIN","math","None","nan","True","False"}:
                    if tok.upper() not in created:
                        inputs.add(tok)
            continue
        # otra sentencia: ignoro (ej. OUTPUT/RETURN generalmente no aparecen)
    # outputs típicos
    out_order = ["EM_EVENTPROBABILITY","EM_CLASSIFICATION"]
    # fallback: tomar la primera P_* como prob
    ret_block = [
        "out = {}",
        "if 'EM_EVENTPROBABILITY' in locals(): out['EM_EVENTPROBABILITY'] = EM_EVENTPROBABILITY",
        "else:",
        "    # fallback: buscar P_*",
        "    cand = [v for v in list(locals().keys()) if v.upper().startswith('P_')]",
        "    if cand: out['EM_EVENTPROBABILITY'] = locals()[cand[0]]",
        "if 'EM_CLASSIFICATION' in locals(): out['EM_CLASSIFICATION'] = EM_CLASSIFICATION",
        "elif 'I_BAD' in locals(): out['EM_CLASSIFICATION'] = I_BAD",
        "return out"
    ]

    header = [
        "import math",
        _src_helpers_block()
    ]
    func_def = [f"def {func_name}(**row):",
                "    # cargar inputs como variables locales",
                "    for k,v in row.items():",
                "        locals()[k] = v",
                "    # --- score code traducido ---"]
    body = ["    "+ln for ln in py_lines] or ["    pass"]
    footer = ["    # --- fin score traducido ---"] + ["    "+ln for ln in ret_block]

    py_code = "\n".join(header + func_def + body + footer)
    exp_inputs = sorted(inputs)
    return py_code, exp_inputs

def _src_helpers_block():
    # helpers que inyectamos en el módulo para que estén disponibles en el score
    return (
        "def MISSING(x):\n"
        "    import math\n"
        "    return x is None or (isinstance(x,float) and math.isnan(x)) or (isinstance(x,str) and x=='')\n"
        "def MEAN(*args):\n"
        "    vals=[float(v) for v in args if not MISSING(v)]\n"
        "    return sum(vals)/len(vals) if vals else float('nan')\n"
        "def SUM(*args):\n"
        "    return sum(float(v) for v in args if not MISSING(v))\n"
        "def NMISS(*args):\n"
        "    return sum(1 for v in args if MISSING(v))\n"
        "def COALESCE(*args):\n"
        "    for v in args:\n"
        "        if not MISSING(v): return v\n"
        "    return None\n"
    )

def compile_sas_score(sas_code: str, func_name="sas_score") -> Tuple[Callable, str, List[str]]:
    """
    Compila el score SAS a una función Python invocable: score_fn(row_dict) -> dict
    Devuelve (score_fn, python_code, expected_inputs).
    """
    py_code, exp_inputs = translate_sas_to_python(sas_code, func_name=func_name)
    module = types.ModuleType("sas_score_generated")
    exec(py_code, module.__dict__)
    score_fn = getattr(module, func_name)
    return score_fn, py_code, exp_inputs

def score_dataframe(score_fn: Callable, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        out = score_fn(**r.to_dict())
        p = float(out.get("EM_EVENTPROBABILITY", list(out.values())[0]))
        i = out.get("EM_CLASSIFICATION", None)
        if i is None:
            i = 1 if p >= threshold else 0
        rows.append({**r.to_dict(), "prob_BAD": p, "label_BAD": int(i) if str(i).isdigit() else (1 if p>=threshold else 0)})
    return pd.DataFrame(rows)
