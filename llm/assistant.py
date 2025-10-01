# llm/assistant.py
# Wrapper para reescritura empática con LLM (OpenAI por defecto)
# - Si no hay OPENAI_API_KEY o falla la llamada, devuelve el texto base.
# - Evita PII: solo usa campos "seguros" del caso (REASON, LOAN).
# - Controla longitud, tono y cumplimiento (no prometer aprobación).

from typing import Dict, Optional, Tuple
import os, math

# OpenAI SDK (opcional). Si no está, degradamos a base text.
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# --------- Utilidades ---------
def _band_from_prob(prob: Optional[float], threshold: float) -> str:
    if prob is None or (isinstance(prob, float) and (prob != prob)):  # NaN
        return "unknown"
    p = float(prob)
    if p < threshold * 0.6:            return "very_low"
    if p < threshold:                  return "near_low"
    if p < min(0.85, threshold + 0.2): return "elevated"
    return "very_high"

def _safe_features(row: Dict) -> Dict:
    """Whitelist de campos: NO PII. Solo lo básico para matizar el texto."""
    out = {}
    # Solo algunas señales útiles (ajustá a tu caso)
    if "REASON" in row and isinstance(row["REASON"], str):
        out["REASON"] = row["REASON"].strip()
    if "_REASON_" in row and isinstance(row["_REASON_"], str) and "REASON" not in out:
        out["REASON"] = row["_REASON_"].strip()
    # LOAN como número
    loan = row.get("LOAN", None)
    try:
        if loan is not None and str(loan).strip() != "":
            out["LOAN"] = int(float(loan))
    except Exception:
        pass
    return out

def _default_client():
    key = os.getenv("OPENAI_API_KEY")
    if not (key and _HAS_OPENAI):
        return None
    try:
        client = OpenAI(api_key=key)
        return client
    except Exception:
        return None

# --------- API pública ---------
def refine_reply_with_llm(
    base_reply: str,
    prob: Optional[float],
    threshold: float,
    row: Optional[Dict] = None,
    temperature: float = 0.6,
    model: str = "gpt-4o-mini"
) -> Tuple[str, str]:
    """
    Devuelve (refined_text, debug_info). Si falla o no hay API key → (base_reply, motivo).
    """
    row = row or {}
    band = _band_from_prob(prob, threshold)
    features = _safe_features(row)

    # Mensaje del sistema: guardrails de tono y compliance
    sys_prompt = (
        "You are a helpful customer support assistant for a lending team. "
        "Rewrite the provided base message to keep it empathetic, concise, and compliant:\n"
        "- Do not guarantee approvals or make commitments.\n"
        "- Keep it under 120 words; avoid jargon and legal advice.\n"
        "- Reflect the risk band and the user's intent, but do not expose internal probabilities.\n"
        "- If helpful, briefly reference the purpose (e.g., Reason) or the amount, without sensitive data.\n"
        "- Maintain a friendly, supportive tone.\n"
    )

    # Mensaje del usuario: contexto mínimo + texto base
    user_prompt = (
        f"Risk band: {band}\n"
        f"Threshold: {threshold:.2f}\n"
        f"Probability available: {'yes' if (prob is not None and not (isinstance(prob, float) and prob != prob)) else 'no'}\n"
        f"Context fields (safe): {features}\n\n"
        f"Base message to rewrite:\n{base_reply}\n\n"
        "Please return only the final rewritten message (no preface, no bullets unless useful for clarity)."
    )

    # Cliente LLM
    client = _default_client()
    if client is None:
        return base_reply, "LLM disabled (no key/SDK). Using base message."

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=float(max(0.0, min(1.0, temperature))),
            max_tokens=180,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return base_reply, "LLM returned empty text. Using base message."
        return text, f"ok [{model}]"
    except Exception as e:
        return base_reply, f"LLM error: {e}"
