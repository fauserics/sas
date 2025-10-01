# sas_code_translator.py
# Translate SAS DATA step scoring code -> Python function.
# With a safe fallback for typical VDMML Logistic code (no indentation blocks).

import re, math
import pandas as pd
from typing import Callable, List, Tuple

# ===== runtime helpers =====
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
        # dot-product loop -> single line (no inner indent)
        m3 = re.match(
            r"\s*do\s+([A-Za-z_]\w*)\s*=\s*1\s*to\s*(\d+)\s*;\s*_linp_\s*\+\s*([A-Za-z_]\w*)\s*\[\s*_\1_\s*\]\s*\*\s*([A-Za-z_]\w*)\s*\[\s*_\1_\s*\]\s*;\s*end\s*;",
            line, flags=re.I
        )
        if m3:
            it, n, a1, a2 = m3.group(1), int(m3.group(2)), m3.group(3), m3.group(4)
            out_lines.append(f"_linp_ = _linp_ + (sum({a1}[i] * {a2}[i] for i in range(1, {n}+1)) if _badval_ == 0 else 0.0)")
            continue
        out_lines.append(line)
    return "\n".join(out_lines)

def preprocess_vdmml_logit_datastep(sas_code: str) -> str:
    s = _sanitize_name_literals(sas_code)
    s = _braces_to_brackets(s)
    s = _arrays_to_python_lists(s)
    # remove labels/goto + decls
    s = re.sub(r"^\s*\w+\s*:\s*$", "", s, flags=re.M)         # labels like skip_123:
    s = re.sub(r"\bgoto\s+\w+\s*;", "", s, flags=re.I)        # remove GOTO
    s = re.sub(r"^\s*(drop|length|label|format|options)\b.*?$", "", s, flags=re.I|re.M)
    return s

# ===== translation to Python body (light; may fail on indent) =====
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
    re_asg  = re.compile(r"^\s*([A-Za-z_]\w*(?:\[[^\]]+\])?)\s*=\s*(.+)$")

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
            created.add(re.sub(r"\[.*?\]","",var).upper())
            py_lines.append("    "*indent + f"{var} = {expr}")
            for tok in re.findall(r"[A-Za-z_]\w*", expr):
                base = tok.upper()
                if base not in {
                    "IF","ELSE","AND","OR","NOT","MISSING","MEAN","SUM","NMISS","COALESCE",
                    "LOG","EXP","ABS","SQRT","MAX","MIN","MATH","NONE","NAN","TRUE","FALSE",
                    "_LINP_","_BADVAL_"
                } and base not in created and not base.startswith("_XROW") and not base.startswith("_BETA"):
                    inputs.add(tok)
            continue
        py_lines.append("    "*indent + st)

    body = "\n".join(py_lines) if py_lines else "pass"
    return body, sorted(inputs)

# ===== Fallback compiler for VDMML Logistic (no indent blocks) =====
def _compile_logistic_fallback(sas_code: str) -> Tuple[Callable, str, List[str]]:
    s = preprocess_vdmml_logit_datastep(sas_code)
    # 1) betas
    m = re.search(r"array\s+(_beta_[A-Za-z0-9_]+)\s*\[\s*(\d+)\s*\]\s*_temporary_\s*\((.*?)\)\s*;", s, flags=re.I|re.S)
    if not m:
        raise SyntaxError("No beta array found for logistic fallback.")
    beta_name, n_str, beta_vals = m.group(1), m.group(2), m.group(3)
    n = int(n_str)
    vals = re.split(r"[, \n\r\t]+", beta_vals.strip())
    vals = [v for v in vals if v != ""]
    if len(vals) != n:
        # algunas versiones separan por espacios; intentamos split por coma
        vals = [v for v in re.split(r"[,\s]+", beta_vals.strip()) if v]
        if len(vals) != n:
            raise SyntaxError("Beta length mismatch.")
    betas = [None] + [float(v) for v in vals]

    # 2) xrow assignments (direct)
    xrow_name = None
    x_assigns = {}  # idx -> expr string
    for idx, expr in re.findall(r"(_xrow_[A-Za-z0-9_]+)\s*\[\s*(\d+)\s*\]\s*=\s*([^;]+);", s):
        xrow_name = xrow_name or idx
        x_assigns[int(expr if idx==xrow_name else idx)] = expr  # noop; fixed below
    # Re-find correctly (typo above), sorry:
    x_assigns = {}
    for name, i_str, expr in re.findall(r"(_xrow_[A-Za-z0-9_]+)\s*\[\s*(\d+)\s*\]\s*=\s*([^;]+);", s):
        xrow_name = xrow_name or name
        x_assigns[int(i_str)] = expr.strip()

    # 3) REASON select -> map dummies
    reason_map = {}  # idx -> literal value
    sel = re.search(r"select\s*\(\s*([A-Za-z_]\w*)\s*\)\s*;(.+?)end\s*;", s, flags=re.I|re.S)
    sel_var = None
    if sel:
        sel_var = sel.group(1)
        block = sel.group(2)
        for cond, stmt in re.findall(r"when\s*\(\s*('.*?')\s*\)\s*([^\;]+)\s*;", block, flags=re.I|re.S):
            mset = re.search(rf"{re.escape(xrow_name)}\s*\[\s*(\d+)\s*\]\s*=\s*_temp_", stmt)
            if mset:
                reason_map[int(mset.group(1))] = cond.strip()

    # 4) missing list (required inputs)
    miss_vars = re.findall(r"missing\(\s*([A-Za-z_][\w']*)\s*\)", s, flags=re.I)
    def san(v):
        m = re.match(r"^'([^']+)'\s*n$", v, flags=re.I)
        return m.group(1) if m else v
    required = sorted({san(v) for v in miss_vars})

    # 5) Build python scorer (no blocks)
    # inputs = union of required + variables referenced on RHS of x_assigns (sans _temp_)
    inputs = set(required)
    for expr in x_assigns.values():
        for tok in re.findall(r"[A-Za-z_]\w*", expr):
            if tok.upper() not in {"_TEMP_", xrow_name.upper(), beta_name.upper(), "IF","THEN","DO","END"}:
                inputs.add(tok)
    if sel_var:
        inputs.add(sel_var)

    inputs = sorted(inputs)

    # Compose python function source (flat)
    lines = []
    lines.append("# fallback logistic scorer (flat, no indents)")
    # init inputs from row
    for v in inputs:
        lines.append(f"{v} = row.get('{v}', None)")
    # numeric cast (soft)
    # we won't force cast; SAS code asume num, pero dejamos al usuario cargar num
    # build x and set defaults
    lines.append(f"x = [0.0]*({n+1})")
    # intercept = 1.0 si está en idx 1
    if 1 in x_assigns and x_assigns[1] in ("1","1.0","1.0 "):
        lines.append("x[1] = 1.0")
    # REASON dummies
    if reason_map and sel_var:
        # set all to 0
        for idx in sorted(reason_map):
            lines.append(f"x[{idx}] = 0.0")
        # set chosen dummy
        for idx, lit in reason_map.items():
            lines.append(f"x[{idx}] = 1.0 if {sel_var} == {lit} else x[{idx}]")
        # bad class if not in any
        lits = " or ".join([f"{sel_var} == {lit}" for lit in reason_map.values()])
        lines.append(f"_badval_ = 0 if ({lits}) else 1")
    else:
        lines.append("_badval_ = 0")
    # direct assignments (skip intercept and REASON ones already handled)
    for idx in sorted(x_assigns):
        if idx == 1 and ("1" in x_assigns[1]): 
            continue
        if idx in reason_map:
            continue
        lines.append(f"x[{idx}] = {x_assigns[idx]}")
    # compute linp
    lines.append("_linp_ = 0.0 if _badval_==1 else sum(x[i]*"+f"{beta_name}[i] for i in range(1,{n+1}))")
    # prob
    lines.append("p1 = (1.0/(1.0+math.exp(-_linp_)) if _linp_>0 else math.exp(_linp_)/(1.0+math.exp(_linp_))) if _badval_==0 else None")
    # output
    lines.append("out = {}")
    lines.append("out['EM_EVENTPROBABILITY'] = p1 if p1 is not None else None")
    lines.append("return out")

    body = "\n".join(lines)

    def _score_fn(**row):
        env = {"math": math, beta_name: [None]+betas[1:]}  # use beta_name var as list
        for k,v in row.items(): env[k]=v
        exec(body, env, env)
        # function body populates 'out' in env and returns it; emulate:
        # but we made 'return out' in body; to capture, re-exec inside wrapper:
        # Simpler: rebuild function and call it
        pass

    # compile proper callable object
    fn_source = "def __scorer__(**row):\n" + "\n".join("    "+ln for ln in body.splitlines())
    loc = {"math": math, beta_name: [None]+betas[1:]}
    exec(fn_source, loc, loc)
    scorer = loc["__scorer__"]

    display_code = "def sas_score(**row):\n" + "\n".join("    "+ln for ln in body.splitlines())
    return scorer, display_code, inputs

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
    pre = preprocess_vdmml_logit_datastep(sas_code)
    body, discovered_inputs = translate_sas_to_python_body(pre)
    inputs = expected_inputs or discovered_inputs or []

    # try the generic exec first
    def _try_compile(body_code: str, inputs_list: list[str]):
        def _score_fn(**row):
            env = {
                "math": math,
                "MISSING": MISSING, "MEAN": MEAN, "SUM": SUM, "NMISS": NMISS, "COALESCE": COALESCE,
                "_linp_": 0.0,
                "_badval_": 0
            }
            for name in inputs_list: env[name] = row.get(name, None)
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

    try:
        score_fn = _try_compile(body, inputs)
        # quick dry run (empty) just to trigger SyntaxError early
        exec(body, {}, {"math": math, "_linp_":0.0, "_badval_":0})
        display_code = (
            "def " + func_name + "(**row):\n"
            "    # inputs available: " + ", ".join(inputs) + "\n"
            "    # --- translated SAS body ---\n" +
            "\n".join(["    " + ln for ln in body.splitlines()]) + "\n" +
            "    # --- end translated body ---\n"
        )
        return score_fn, display_code, inputs
    except SyntaxError:
        # fallback for logistic (flat, no indents)
        scorer, display_code, fb_inputs = _compile_logistic_fallback(sas_code)
        return scorer, display_code, (expected_inputs or fb_inputs)

# ===== DataFrame helper =====
def score_dataframe(score_fn: Callable, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        out = score_fn(**r.to_dict()) or {}
        p = out.get("EM_EVENTPROBABILITY", None)
        if p is None:
            raise ValueError("Translated score returned no probability. Check SAS code or fallback parser.")
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
