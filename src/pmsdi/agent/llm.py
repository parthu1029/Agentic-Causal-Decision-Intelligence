"""OPTIONAL natural-language polishing of the evidence summary / explanation.
The numbers always come from the models; the LLM may only rephrase them.  Disabled unless ANTHROPIC_API_KEY is set
(not exercised in the test-suite).  Set PMSDI_LLM_MODEL to choose a model."""
from __future__ import annotations

import os


def enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def polish(text: str, role: str = "evidence summary") -> str:
    if not enabled():
        return text
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=os.environ.get("PMSDI_LLM_MODEL", "claude-sonnet-5-5"), max_tokens=400,
            messages=[{"role": "user", "content": (
                f"Rewrite this property-management {role} for a busy manager in 4 short sentences. "
                "Do NOT add, remove or change any number or action name.\n\n" + text)}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text") or text
    except Exception:
        return text
