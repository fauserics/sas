# sas_code_translator.py
# Minimal hotfix: robust set unions + API estable usada por la app.
# (No cambia la lógica de scoring; solo evita "unsupported operand type(s) for |: 'list' and 'set'".)

from __future__ import annotations
import re
import math
import json
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import pandas as pd
import numpy as np

# ------------------------- helpers -------------------------

def _as_set(x: Any) -> Set[Any]:
    """Convierte x a set de forma segura."""
    if x is None:
        return set()
    if isinstance(x, set):
        return x
    if isinstance(x, (list, tuple)):
        return set(x)
    return {x}

def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

# ------------------------- extracción simple -------------------------

_re_select_reason = re.compile(r"select\s*\(\s*_?REASON_?\s*\)\s*;", re.I)
_re_when_str = re.compile(r"when\s*\(\s*'([^']+)'\s*\)", re.I)
_re_default_prob = re.compile(r"""if\s+"?P_?BAD"?\s*=\s*\.\s*then\s+"?P_?BAD"?\s*=\s*([0-9eE\.\-\+]+)""", re.I)

def extract_categorical_levels(sas_code: str) -> Dict[str, Set[str]]:
    """
    Extrae niveles de categorías más comunes (e.g., REASON) del score SAS (DATA step).
    Devuelve dict var -> set(levels).
    """
    cats: Dict[str, Set[str]] = {}
    code = sas_code or ""
    lines = code.splitlines()

    # REASON via select/when
    for i, ln in enumerate(lines):
        if _re_select_reason.search(ln):
            levels: Set[str] = set()
            j = i + 1
            while j < len(lines):
                m = _re_when_str.search(lines[j])
                if m:
                    levels.add(m.group(1))
                if re.search(r"\bend\s*;", lines[j], re.I):
                    break
                j += 1
            if levels:
                cats["REASON"] = levels
            break

    # Otras categorías se pueden agregar aquí si hace falta
    return cats

def extract_default_prob_hint(sas_code: str) -> Optional[float]:
    """Detecta un valor default de P_BAD (si viene en el code DS2/VA)."""
    m = _re_default_prob.search(sas_code or "")
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

# ------------------------- "Compilación" (liviana) -------------------------

def compile_sas_score(sas_code: str,
                      known_categories: Optional[Dict[str, Iterable[str]]] = None) -> Dict[str, Any]:
    """
    Devuelve:
      - score_fn: callable(df: pd.DataFrame, threshold: float=0.5) -> pd.DataFrame
      - categories: Dict[var, List[level]]
      - default_prob: Optional[float]
    NOTA: Esta función no implementa la traducción completa del modelo; si el code incluye
          P_BAD / EM_EVENTPROBABILITY, se usa; si no, devuelve NaN (como antes).
    """
    # 1) Extraer categorías del SAS
    parsed_cats = extract_categorical_levels(sas_code)
    # 2) Merge con conocidas (pueden venir como list/tuple desde inputVar.json)
    merged: Dict[str, Set[str]] = {}
    known_categories = known_categories or {}
    for var in set(list(parsed_cats.keys()) + list(known_categories.keys())):
        left = _as_set(parsed_cats.get(var))
        right = _as_set(known_categories.get(var))
        merged[var] = left | right  # <-- ahora ambas son set

    categories_out: Dict[str, List[str]] = {k: sorted(list(v)) for k, v in merged.items()}
    default_prob = extract_default_prob_hint(sas_code)

    # 3) Construir score_fn liviano:
    #    - Si existe columna 'P_BAD' o 'EM_EVENTPROBABILITY' en la salida simulada, la usamos.
    #    - Como estamos traduciendo un DATA step heterogéneo, mantenemos la heurística:
    #      * Si no hay forma de calcular, devolvemos NaN (igual que antes).
    prob_keys = ("EM_EVENTPROBABILITY", "P_BAD", "P_va__d__E_JOB1", "P_bad1", "P_TARGET1")

    def _score_row_like_sas(row: Dict[str, Any]) -> float:
        # Heurística mínima: si el caller ya pasó una prob (por ejemplo, post-scorer),
        # o si la app nos da un dict con alguno de los prob_keys, úsala.
        for k in prob_keys:
            v = row.get(k)
            if v is not None and str(v).strip() != "":
                return _safe_float(v)
        # Si hay pista de default_prob del code, úsala como fallback
        if default_prob is not None:
            return float(default_prob)
        return float("nan")

    def score_dataframe(score_fn_compiled, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        # score_fn_compiled se ignora aquí para mantener compat con la firma existente en app.py
        probs = []
        for _, r in df.iterrows():
            probs.append(_score_row_like_sas(r.to_dict()))
        out = df.copy()
        out["Probability"] = probs
        out["Predicted"] = (pd.Series(probs) >= float(threshold)).astype(int)
        return out

    # Devolvemos algo estructuralmente similar a lo que espera la app:
    return {
        "score_fn": _score_row_like_sas,     # por compat, aunque no se usa internamente aquí
        "categories": categories_out,         # <- ahora siempre listas, no sets
        "default_prob": default_prob,
        "score_dataframe": score_dataframe
    }

# Para compatibilidad con import en app.py:
def score_dataframe(score_fn, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Wrapper de compatibilidad: llama a compile_sas_score(...)? No.
    La app nos pasa (score_fn, df, threshold). Aquí delegamos a una
    versión que no depende de score_fn (por backward-compat).
    """
    probs = []
    for _, r in df.iterrows():
        # si score_fn es callable, úsalo; si no, heurística
        if callable(score_fn):
            try:
                p = score_fn(r.to_dict())
            except Exception:
                p = float("nan")
        else:
            p = float("nan")
        probs.append(_safe_float(p))
    out = df.copy()
    out["Probability"] = probs
    out["Predicted"] = (pd.Series(probs) >= float(threshold)).astype(int)
    return out
