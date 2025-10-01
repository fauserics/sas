# sas_code_translator.py
# Translate SAS DATA step scoring code -> Python function.
# Fallback logístico tolerante a faltantes (si faltan valores, usa 0.0 / sin dummies).

import re, math
import pandas as pd
from typing import Callable, List, Tuple

# ===== runtime helpers =====
def MISSING(x):
    return x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and x.strip() == "")

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

def _to_num(x):
    try:
        return float(x)
    except Exception:
        return 0.0

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
    stmt = re.sub(r"\bEQ\b", "==", stmt, flags=re.I)
    stmt = re.sub(r"\bNE\b", "!=", stmt, flags=re.I)
    stmt = re.sub(r"\bGE\b", ">=", stmt, flags=re.I)
    stmt = re.sub(r"\bLE\b", "<=", stmt, flags=re.I)
    stmt = re.sub(r"\bGT\b", ">", stmt, flags=re.I)
    stmt = re.sub(r"\bLT\b", "<", stmt, flags=re.I)
    stmt = re.sub(r"\bAND\b", "and", stmt, flags=re.I)
    stmt = re.sub(r"\bOR\b", "or", stmt, flags=re.I)
    stmt = re.sub(r"\bNOT\b", "not", stmt, flags=re.I)
    stmt = re.sub(r"\bLOG\(", "math.log(", stmt, flags=re.I)
    stmt = re.sub(r"\bEXP\(", "math.exp(", stmt, flags=re.I)
    stmt = re.sub(r"\bABS\(", "abs(", stmt, flags=re.I)
    stmt = re.sub(r"\bSQRT\(", "math.sqrt(", stmt, flags=re.I)
    stmt = re.sub(r"\bMAX\(", "max(", stmt, flags=re.I)
    stmt = re.sub(r"\bMIN\(", "min(", stmt, flags=re.I)
    stmt = re.sub(r"\bMEAN\(", "MEAN(", stmt, flags=re.I)
    stmt = re.sub(r"\bSUM\(", "SUM(", stmt, flags=re.I)
    stmt = re.sub(r"\bNMISS\(", "NMISS(", stmt, flags=re.I)
    stmt = re.sub(r"\bCOALESCE\(", "COALESCE(", stmt, flags=re.I)
    if re.match(r"^\s*if\b", stmt, flags=re.I):
        stmt = re.sub(r"(?<![<>=!])=(?![=])", "==", stmt)
    return stmt.strip()

# ===== preprocessing for VDMML Logistic DATA step =====
def _sanitize_name_literals(txt: str) -> str:
    def repl(m):
        raw = m.group(1)
        s = re.sub(r"\W", "_", raw)
        return s if s else "_Q_"
    return re.sub(r"'([^']+?)'\s*n", repl, txt, flags=re.I)

def _braces_to_brackets(txt: str) -> str:
    txt = re.sub(r"\{", "[", txt)
    txt = re.sub(r"\}", "]", txt)
    return txt

def _arrays_to_python_lists(txt: str) -> str:
    out_lines = []
    for line in txt.splitlines():
        m = re.match(r"\s*array\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*_temporary_\s*\((.*?)\)\s*;", line, flags=re.I|re.S)
        if m:
            name, n, vals = m.group(1), int(m.group(2)), m.group(3).strip()
            out_lines.append(f"{name} = [None] + [{vals}]")
            continue
        m2 = re.match(r"\s*array\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*_temporary_\s*;", line, flags=re.I)
        if m2:
            name, n = m2.group(1), int(m2.group(2))
            out_lines.append(f"{name} = [0.0] * ({n}+1)")
            continue
        m3 = re.match(
            r"\s*do\s+[A-Za-z_]\w*\s*=\s*1\s*to\s*(\d+)\s*;\s*_linp_\s*\+\s*([A-Za-z_]\w*)\s*\[\s*_[A-Za-z_]\w*_\s*\]\s*\*\s*([A-Za-z_]\w*)\s*\[\s*_[A-Za-z_]\w*_\s*\]\s*;\s*end\s*;",
            line, flags=re.I
        )
        if m3:
            n, a1, a2 = int(m3.group(1)), m3.group(2), m3.group(3)
            out_lines.append(f"_linp_ = _linp_ + (sum({a1}[i] * {a2}[i] for i in range(1, {n}+1)) if _badval_ == 0 else 0.0)")
            continue
        out_lines.append(line)
    return "\n".join(out_lines)

def preprocess_vdmml_logit_datastep(sas_code: str) -> str:
    s = _sanitize_name_literals(sas_code)
    s = _braces_to_brackets(s)
    s = _arrays_to_python_lists(s)
    s = re.sub(r"^\s*\w+\s*:\s*$", "", s, flags=re.M)
    s = re.sub(r"\bgoto\s+\w+\s*;", "", s, flags=re.I)
    s = re.sub(r"^\s*(drop|length|label|format|options)\b.*?$", "", s, flags=re.I|re.M)
    return s

# ===== generic translator (puede fallar por indent) =====
def translate_sas_to_python_body(sas_code: str) -> Tuple[str, List[str]]:
    code = _strip_comments(sas_code)
    stmts = _split_statements(code)
    py_lines, inputs, created = [], set(), set()
    indent = 0

    re_if   = re.compile(r"^\s*if\b(.*)\bthen\b\s*(do)?\s*$", flags=re.I)
    re_else = re.compile(r"^\s*else\b\s*(do)?\s*$", flags=re.I)
    re_end  = re.compile(r"^\s*end\s*$", flags=re.I)
    re_asg  = re.compile(r"^\s*([A-Za-z_]\w*(?:\[[^\]]+\])?)\s*=\s*(.+)$")

    for raw in stmts:
        st = _clean_stmt(raw)
        if not st: continue
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
            created.add(re.sub(r"\[.*?\]","",var).upper())
            py_lines.append("    "*indent + f"{var} = {expr}")
            for tok in re.findall(r"[A-Za-z_]\w*", expr):
                U = tok.upper()
                if U not in {
                    "IF","ELSE","AND","OR","NOT","MISSING","MEAN","SUM","NMISS","COALESCE",
                    "LOG","EXP","ABS","SQRT","MAX","MIN","MATH","NONE","NAN","TRUE","FALSE",
                    "_LINP_","_BADVAL_"
                } and U not in created and not U.startswith("_XROW") and not U.startswith("_BETA"):
                    inputs.add(tok)
            continue
        py_lines.append("    "*indent + st)
    body = "\n".join(py_lines) if py_lines else "pass"
    return body, sorted(inputs)

# ===== logistic fallback (tolerante a faltantes) =====
def _compile_logistic_fallback(sas_code: str) -> Tuple[Callable, str, List[str]]:
    s = preprocess_vdmml_logit_datastep(sas_code)

    m = re.search(r"array\s+(_beta_[A-Za-z0-9_]+)\s*\[\s*(\d+)\s*\]\s*_temporary_\s*\((.*?)\)\s*;", s, flags=re.I|re.S)
    if not m:
        raise SyntaxError("No beta array found for logistic fallback.")
    n = int(m.group(2))
    toks = [t for t in re.split(r"[,\s]+", m.group(3).strip()) if t]
    if len(toks) != n:
        raise SyntaxError("Beta length mismatch.")
    beta = [None] + [float(t) for t in toks]

    xrow_name = None
    x_assigns = {}
    for name, i_str, expr in re.findall(r"(_xrow_[A-Za-z0-9_]+)\s*\[\s*(\d+)\s*\]\s*=\s*([^;]+);", s, flags=re.I|re.S):
        xrow_name = xrow_name or name
        x_assigns[int(i_str)] = expr.strip()

    reason_map, sel_var = {}, None
    sel = re.search(r"select\s*\(\s*([A-Za-z_]\w*)\s*\)\s*;(.+?)end\s*;", s, flags=re.I|re.S)
    if sel:
        sel_var = sel.group(1).strip()
        block = sel.group(2)
        for cond, stmt in re.findall(r"when\s*\(\s*('.*?')\s*\)\s*([^\;]+)\s*;", block, flags=re.I|re.S):
            mset = re.search(rf"{re.escape(xrow_name)}\s*\[\s*(\d+)\s*\]\s*=\s*_temp_", stmt, flags=re.I)
            if mset:
                reason_map[int(mset.group(1))] = cond.strip()

    # requeridas (pero NO cortamos si faltan)
    miss_vars = re.findall(r"missing\(\s*([A-Za-z_][\w']*)\s*\)", s, flags=re.I)
    def san(v):
        m = re.match(r"^'([^']+)'\s*n$", v, flags=re.I)
        return m.group(1) if m else v
    required = sorted({san(v) for v in miss_vars if san(v).upper() != "BAD"})

    # inputs potenciales
    inputs = set(required)
    for expr in x_assigns.values():
        for tok in re.findall(r"[A-Za-z_]\w*", expr):
            U = tok.upper()
            if U not in {"_TEMP_", "MISSING", "MEAN", "NMISS", "SUM"} and not U.startswith("_XROW") and not U.startswith("_BETA"):
                inputs.add(tok)
    if sel_var: inputs.add(sel_var)
    inputs = sorted(inputs)

    # construir scorer tolerante
    body = []
    body.append("# fallback logistic scorer (tolerant to missing)")
    # cargar entradas (si falta -> None)
    for v in inputs:
        body.append(f"{v} = row.get('{v}', None)")
    # construir x
    body.append(f"x = [0.0]*({n+1})")
    if 1 in x_assigns and re.fullmatch(r"1(\.0+)?", x_assigns[1]):
        body.append("x[1] = 1.0")
    # dummies por select/when
    if reason_map and sel_var:
        for idx in sorted(reason_map):
            body.append(f"x[{idx}] = 1.0 if {sel_var} == {reason_map[idx]} else 0.0")
    # resto de columnas
    for idx in sorted(x_assigns):
        if idx == 1 and re.fullmatch(r"1(\.0+)?", x_assigns[1]):  # intercepto
            continue
        if idx in reason_map:
            continue
        expr = x_assigns[idx].strip()
        if re.fullmatch(r"\d+(\.\d+)?", expr):
            body.append(f"x[{idx}] = float({expr})")
        else:
            body.append(f"x[{idx}] = _to_num({expr})")

    body.append(f"BETA = {beta}")
    body.append(f"_linp_ = sum(x[i]*BETA[i] for i in range(1, {n+1}))")
    body.append("p1 = (1.0/(1.0+math.exp(-_linp_)) if _linp_>0 else math.exp(_linp_)/(1.0+math.exp(_linp_)))")
    body.append("return {'EM_EVENTPROBABILITY': p1}")

    fn_src = "def __scorer__(**row):\n" + "\n".join("    "+ln for ln in body)
    loc = {"math": math, "_to_num": _to_num}
    exec(fn_src, loc, loc)
    scorer = loc["__scorer__"]

    display_code = "def sas_score(**row):\n" + "\n".join("    "+ln for ln in body)
    return scorer, display_code, inputs

# ===== compile to callable =====
def _pick_prob_from_env(env: dict) -> float | None:
    for k in ("EM_EVENTPROBABILITY","EM_PREDICTION"):
        if k in env and env[k] is not None:
            try: return float(env[k])
            except Exception: pass
    p_keys = [k for k in env.keys() if k.upper().startswith("P_") and env[k] is not None]
    if not p_keys: return None
    def key_score(k):
        ku = k.upper(); score = 0
        if ku.endswith("1"): score += 3
        if "BAD1" in ku or "EVENT" in ku: score += 2
        if ku in ("P_BAD","P_TARGET1"): score += 1
        return score
    p_keys.sort(key=lambda k: (key_score(k), k), reverse=True)
    try: return float(env[p_keys[0]])
    except Exception: return None

def compile_sas_score(sas_code: str, func_name: str = "sas_score", expected_inputs: list[str] | None = None
                      ) -> Tuple[Callable, str, List[str]]:
    pre = preprocess_vdmml_logit_datastep(sas_code)
    body, discovered_inputs = translate_sas_to_python_body(pre)

    merged_inputs = sorted(set(expected_inputs or []) | set(discovered_inputs or []))

    # Si el código menciona missing(BAD) o falla el exec del cuerpo traducido => fallback tolerante
    try:
        exec(body, {}, {"math": math, "_linp_":0.0, "_badval_":0})
        score_fn = None
        display_code = (
            "def " + func_name + "(**row):\n"
            "    # inputs available: " + ", ".join(merged_inputs) + "\n"
            "    # --- translated SAS body ---\n" +
            "\n".join(["    " + ln for ln in body.splitlines()]) + "\n" +
            "    # --- end translated body ---\n"
        )
        score_fn = _build_exec_wrapper(body, merged_inputs)
        # Si hay BAD en missing(), prefiero fallback para evitar NaN/None
        if re.search(r"missing\(\s*'?BAD'?\s*\)", pre, flags=re.I):
            raise RuntimeError("Force fallback due to missing(BAD)")
        return score_fn, display_code, merged_inputs
    except Exception:
        scorer, display_code, fb_inputs = _compile_logistic_fallback(sas_code)
        return scorer, display_code, (expected_inputs or fb_inputs)

def _build_exec_wrapper(body_code: str, inputs_list: list[str]):
    def _score_fn(**row):
        env = {
            "math": math,
            "MISSING": MISSING, "MEAN": MEAN, "SUM": SUM, "NMISS": NMISS, "COALESCE": COALESCE,
            "_linp_": 0.0,
            "_badval_": 0
        }
        for name in inputs_list:
            env[name] = row.get(name, None)
        for k, v in row.items():
            if k not in env: env[k] = v
        exec(body_code, {}, env)
        out = {}
        p = _pick_prob_from_env(env)
        if p is not None:
            out["EM_EVENTPROBABILITY"] = p
        for key in ("EM_CLASSIFICATION","I_BAD","I_TARGET","LABEL"):
            if key in env and env[key] is not None:
                out["EM_CLASSIFICATION"] = env[key]; break
        return out
    return _score_fn

# ===== DataFrame helper =====
def score_dataframe(score_fn: Callable, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        out = score_fn(**r.to_dict()) or {}
        p = out.get("EM_EVENTPROBABILITY", None)
        if p is None or (isinstance(p, float) and (math.isnan(p))):
            p = float("nan"); lab = None
        else:
            p = float(p)
            lab = 1 if p >= threshold else 0
        rec = r.to_dict()
        rec["prob_BAD"] = p
        rec["label_BAD"] = (int(lab) if lab is not None else None)
        rows.append(rec)
    return pd.DataFrame(rows)
