"""Guardrails for untrusted tool output.

Threats handled (OWASP Top 10 for LLM applications):
* LLM01 indirect prompt injection: logs can contain attacker text such as
  "ignore previous instructions". We detect it, flag the evidence, and wrap tool output
  in <untrusted_data> tags that the system prompt tells the model never to obey.
* LLM02 sensitive information disclosure: secrets and PII are redacted before any text
  is sent to the model.
* LLM06 excessive agency: side-effect tools are blocked and surfaced for human approval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_INJECTION_PATTERNS = [
    r"ignore (?:all |any )?(?:previous|prior|above|earlier) (?:instructions|prompts|rules)",
    r"disregard (?:the |your )?(?:system|previous|prior) (?:prompt|instructions)",
    r"you are now (?:a|an|in)\b",
    r"(?:reveal|print|show) (?:your|the) (?:system )?(?:prompt|instructions)",
    r"new instructions?:",
    r"\b(?:call|use|run|invoke) (?:the )?(?:tool )?rollback_deploy\b",
    r"curl\s+https?://",
    r"base64\s+-d",
]
_INJECTION = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

_SECRETS: dict[str, re.Pattern[str]] = {
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "bearer_token": re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/-]{20,}=*"),
    "password": re.compile(r"(?i)\b(password|passwd|pwd)\s*[=:]\s*\S+"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
}
_CARD_CANDIDATE = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")

MAX_TOOL_OUTPUT_CHARS = 4000


@dataclass(frozen=True)
class ScreenResult:
    text: str
    injection_signals: list[str]
    redactions: dict[str, int]

    @property
    def flagged(self) -> bool:
        return bool(self.injection_signals)


def detect_injection(text: str) -> list[str]:
    return [p.pattern for p in _INJECTION if p.search(text)]


def luhn_valid(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        checksum += d
    return len(digits) >= 13 and checksum % 10 == 0


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for label, pattern in _SECRETS.items():
        text, n = pattern.subn(f"[REDACTED:{label}]", text)
        if n:
            counts[label] = n

    # Luhn check keeps request IDs and epoch-millisecond timestamps from being redacted.
    cards = 0

    def replace_card(match: re.Match[str]) -> str:
        nonlocal cards
        if luhn_valid(match.group(0)):
            cards += 1
            return "[REDACTED:card_number]"
        return match.group(0)

    text = _CARD_CANDIDATE.sub(replace_card, text)
    if cards:
        counts["card_number"] = cards
    return text, counts


def screen_tool_output(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> ScreenResult:
    signals = detect_injection(text)
    clean, counts = redact(text)
    if len(clean) > max_chars:
        clean = clean[:max_chars] + f"\n...[truncated {len(clean) - max_chars} chars]"
    return ScreenResult(clean, signals, counts)


def wrap_untrusted(source: str, text: str, flagged: bool) -> str:
    # Neutralize any closing tag inside the data so it cannot break out of the wrapper.
    safe = text.replace("</untrusted_data>", "</untrusted_data_>")
    flag = "true" if flagged else "false"
    header = f'<untrusted_data source="{source}" injection_suspected="{flag}">'
    return f"{header}\n{safe}\n</untrusted_data>"
