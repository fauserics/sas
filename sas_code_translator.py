# sas_code_translator.py
# Translate SAS DATA step scoring code -> Python function.
# Also includes preprocessing for typical VDMML Logistic code patterns.

import re, math, types
import pandas as pd
from typing import Callable, List, Tuple

# ===== runtime helpers available to translated code =====
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
        if not MISSING(v): return v
    return None

# ===== mappings =====
_OP_MAP = {
    r"\bEQ\b": "==", r"\bNE\b": "!=", r"\bGE\b": ">=", r"\bLE\b": "<=",
    r"\bGT\b": ">",  r"\bLT\b": "<",  r"\bAND\b": "and", r"\bOR\b": "or", r"\bNOT\b": "not"
}
_FUNC_MAP = {
    r"\bLOG\(": "math.log(", r"\bEXP\(": "math.exp(", r"\bABS\(": "abs(",
    r"\bSQRT\(": "math.sqrt(", r"\bMAX\(": "max(", r"\bMIN\(": "min(",
    r"\bMEAN\(": "MEAN(", r"\bSUM\(": "SUM(", r"\bNMISS\(": "NMISS(", r"\bCOALESCE\(": "COALESCE("
}

# ===== core text utils =====
def _strip_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"^\s*\*.*?;\s*$", "", s, flags=re.M)
    return s

def _split_statements(sas_code: str) -> List[str]:
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
    if re.match(r"^\s*(label|length|format|drop|keep|dcl|declare|retain|array|return|options)\b", stmt, flags=re.I):
        return ""
    if re.match(r"^\s*(package|method|enddata|run)\b", stmt, flags=re.I):
        return ""
    stmt = stmt.replace("||", "+")
    stmt = re.sub(r"=\s*\.", "= None", stmt)
    for k,v in _OP_MAP.items():   stmt = re.sub(k, v, stmt, flags=re.I)
    for k,v in _FUNC_MAP.items(): stmt = re.sub(k, v, stmt, flags=re.I)
    if re.match(r"^\s*if\b", stmt, flags=re.I):
        stmt = re.sub(r"(?<![<>=!])=(?![=])", "==", stmt)
    return stmt.strip()

# ===== preprocessing for VDMML Logistic DATA step =====
def _sanitize_name_literals(txt: str) -> str:
    # 'P_X'n -> P_X
    def repl(m):
        raw = m.group(1)
        s = re.sub(r"\W", "_", raw)
        return s if s else "_Q_"
    return re.sub(r"'([^']+?)'\s*n", repl, txt, flags=re.I)

def _arrays_to_python_lists(txt: str) -> str:
    out_lines = []
    for line in txt.splitlines():
        m = re.match(r"\s*array\s+([A-Za-z_]\w*)\s*\{\s*(\d+)\s*\}\s*_temporary_\s*\((.*?)\)\s*;", line, flags=re.I|re.S)
        if m:
            name, n, vals = m.group(1), int(m.group(2)), m.group(3)
            vals_str = vals.strip()
            py = f"{name} = [None] + [{vals_str}]"
            out_lines.append(py)
            continue
        m2 = re.match(r"\s*array\s+([A-Za-z_]\w*)\s*\{\s*(\d+)\s*\}\s*_temporary_\s*;", line, flags=re.I)
        if m2:
            name, n = m2.group(1), int(m2.group(2))
            py = f"{name} = [0.0] * ({n}+1)"
            out_lines.append(py)
            continue
        # zeroing loop: do _i_=1 to N; arr[_i_] = 0; end;
        mzero = re.match(r"\s*do\s+([A-Za-z_]\w*)\s*=\s*1\s*to\s*(\d+)\s*;\s*([A-Za-z_]\w*)\s*\{\s*_\1_\s*\}\s*=\s*0\s*;\s*end\s*;", line, flags=re.I)
        if mzero:
            # already initialized to zeros; skip
            out_lines.append("# init zeros (removed)")
            continue
        # dot-product loop: do _i_=1 to N; _linp_ + arr1{_i_} * arr2{_i_}; end;
        m3 = re.match(
            r"\s*do\s+([A-Za-z_]\w*)\s*=\s*1\s*to\s*(\d+)\s*;\s*_linp_\s*\+\s*([A-Za-z_]\w*)\s*\{\s*_\1_\s*\}\s*\*\s*([A-Za-z_]\w*)\s*\{\s*_\1_\s*\}\s*;\s*end\s*;",
            line, flags=re.I
        )
        if m3:
            it, n, a1, a2 = m3.group(1), int(m3.group(2)), m3.group(3), m3.group(4)
            py = f"if _badval_ == 0:\n    _linp_ = _linp_ + sum({a1}[i] * {a2}[i] for i in range(1, {n}+1))"
            out_lines.append(py)
            continue
        out_lines.append(line)
    return "\n".join(out_lines)

def _select_when_to_if(txt: str) -> str:
    lines = _split_statements(txt)
    res, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        msel = re.match(r"^select\s*\((.+)\)$", s, flags=re.I)
        if not msel:
            res.append(lines[i]); i += 1; continue
        sel_expr = msel.group(1).strip()
        i += 1; first = True
        while i < len(lines):
            w = lines[i].strip()
            if re.match(r"^end$", w, flags=re.I):
                i += 1; break
            mwhen = re.match(r"^when\s*\((.+)\)\s*(.+)$", w, flags=re.I)
            moth  = re.match(r"^otherwise\s*(.+)$", w, flags=re.I)
            if mwhen:
                cond = mwhen.group(1).strip()
                stmt = mwhen.group(2).strip()
                res.append((("if " if first else "elif ") + f"{sel_expr} == {cond}:"))
                res.append("    " + stmt)
                first = False; i += 1; continue
            if moth:
                stmt = moth.group(1).strip()
                res.append("else:"); res.append("    " + stmt)
                i += 1; continue
            res.append("    " + lines[i]); i += 1
        continue
    return ";\n".join(res)

def _braces_to_brackets(txt: str) -> str:
    txt = re.sub(r"\{", "[", txt)
    txt = re.sub(r"\}", "]", txt)
    return txt

def preprocess_vdmml_logit_datastep(sas_code: str) -> str:
    s = sas_code
    s = _sanitize_name_literals(s)
    s = _arrays_to_python_lists(s)
    s = _select_when_to_if(s)
    s = _braces_to_brackets(s)
    # remove labels/goto (we keep _badval_ gating)
    s = re.sub(r"^\s*\w+\s*:\s*$", "", s, flags=re.M)         # labels like skip_123:
    s = re.sub(r"\bgoto\s+\w+\s*;", "", s, flags=re.I)        # remove GOTO
    s = re.sub(r"^\s*(drop|length|label|format|options)\b.*?$", "", s, flags=re.I|re.M)
    return s

# ===== translation to Python body =====
def translate_sas_to_python_body(sas_code: str) -> Tuple[str, List[str]]:
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
            indent += 1; continue
        if re_else.match(st):
            indent = max(0, indent-1)
            py_lines.append("    "*indent + "else:")
            indent += 1; continue
        m_as = re_asg.match(st)
        if m_as:
            var, expr = m_as.group(1), m_as.group(2)
            created.add(var.upper())
            py_lines.append("    "*indent + f"{var} = {expr}")
            for tok in re.findall(r"[A-Za-z_]\w*", expr):
                if tok.upper() not in {
                    "IF","ELSE","AND","OR","NOT","MISSING","MEAN","SUM","NMISS","COALESCE",
                    "LOG","EXP","ABS","SQRT","MAX","MIN","MATH","NONE","NAN","TRUE","FALSE",
                    "_LINP_","_BADVAL_"
                } and tok.upper() not in created:
                    inputs.add(tok)
            continue
        py_lines.append("    "*indent + st)

    body = "\n".join(py_lines) if py_lines else "pass"
    return body, sorted(inputs)

# ===== compile to callable =====
def _pick_prob_from_env(env: dict) -> float | None:
    for k in ("EM_EVENTPROBABILITY","EM_PREDICTION"):
        if k in env and env[k] is not None:
            try: return float(env[k])
            except Exception: pass
    p_keys = [k for k in env.keys() if k.upper().startswith("P_") and env[k] is not None]
    if not p_keys:
        return None
    def key_score(k):
        ku = k.upper(); score = 0
        if ku.endswith("1"): score += 3
        if "BAD1" in ku or "EVENT" in ku: score += 2
        if ku in ("P_BAD","P_TARGET1"): score += 1
        return score
    p_keys.sort(key=lambda k: (key_score(k), k), reverse=True)
    try:
        return float(env[p_keys[0]])
    except Exception:
        return None

def compile_sas_score(sas_code: str, func_name: str = "sas_score", expected_inputs: list[str] | None = None
                      ) -> Tuple[Callable, str, List[str]]:
    sas_code = preprocess_vdmml_logit_datastep(sas_code)
    body, discovered_inputs = translate_sas_to_python_body(sas_code)
    inputs = expected_inputs or discovered_inputs or []

    def _score_fn(**row):
        env = {
            "math": math,
            "MISSING": MISSING, "MEAN": MEAN, "SUM": SUM, "NMISS": NMISS, "COALESCE": COALESCE,
            "_linp_": 0.0,
            "_badval_": 0
        }
        for name in inputs: env[name] = row.get(name, None)
        for k, v in row.items():
            if k not in env: env[k] = v
        exec(body, {}, env)
        out = {}
        p = _pick_prob_from_env(env)
        if p is not None:
            out["EM_EVENTPROBABILITY"] = p
        for key in ("EM_CLASSIFICATION","I_BAD","I_TARGET","LABEL"):
            if key in env and env[key] is not None:
                out["EM_CLASSIFICATION"] = env[key]; break
        return out

    display_code = (
        "def " + func_name + "(**row):\n"
        "    # inputs available: " + ", ".join(inputs) + "\n"
        "    # --- translated SAS body ---\n" +
        "\n".join(["    " + ln for ln in body.splitlines()]) + "\n" +
        "    # --- end translated body ---\n"
        "    # runtime builds EM_EVENTPROBABILITY from EM_PREDICTION or P_* if needed\n"
    )
    return _score_fn, display_code, inputs

# ===== DataFrame helper =====
def score_dataframe(score_fn: Callable, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        out = score_fn(**r.to_dict()) or {}
        p = out.get("EM_EVENTPROBABILITY", None)
        if p is None:
            raise ValueError("Translated score returned no probability. Check that SAS code assigns EM_PREDICTION/EM_EVENTPROBABILITY or a P_* variable.")
        p = float(p)
        lab = out.get("EM_CLASSIFICATION", None)
        if lab is None:
            lab = 1 if p >= threshold else 0
        else:
            try: lab = int(lab)
            except Exception: lab = 1 if p >= threshold else 0
        rec = r.to_dict(); rec["prob_BAD"] = p; rec["label_BAD"] = int(lab)
        rows.append(rec)
    return pd.DataFrame(rows)
