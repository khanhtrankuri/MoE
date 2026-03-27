from __future__ import annotations

import re
import unicodedata

try:
    import regex as regex_mod
except ImportError:
    regex_mod = None


_WHITESPACE_RE = re.compile(r"\s+")
_TELUGU_START = 0x0C00
_TELUGU_END = 0x0C7F


def normalize_transcript(
    value,
    *,
    unicode_form: str = "NFC",
    collapse_whitespace: bool = True,
    strip_bom: bool = True,
) -> str:
    text = "" if value is None else str(value)
    if strip_bom:
        text = text.replace("\ufeff", "")
    if unicode_form:
        text = unicodedata.normalize(unicode_form.upper(), text)
    if collapse_whitespace:
        text = _WHITESPACE_RE.sub(" ", text).strip()
    else:
        text = text.strip()
    return text


def split_graphemes(text: str) -> list[str]:
    if regex_mod is None:
        raise RuntimeError(
            "The 'regex' package is required for grapheme tokenization. "
            "Install dependencies with 'pip install -r requirements.txt'."
        )
    return regex_mod.findall(r"\X", text)


def is_telugu_script_char(ch: str) -> bool:
    if not ch:
        return False
    codepoint = ord(ch)
    return _TELUGU_START <= codepoint <= _TELUGU_END


def collect_out_of_script_chars(text: str) -> list[str]:
    flagged: list[str] = []
    seen: set[str] = set()
    for ch in text:
        if ch.isspace() or is_telugu_script_char(ch):
            continue
        category = unicodedata.category(ch)
        if category.startswith(("P", "S", "N")):
            continue
        if ch not in seen:
            flagged.append(ch)
            seen.add(ch)
    return flagged


def preview_text(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."
