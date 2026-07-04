import ast
import math
import re

# Value may contain balanced brackets one level deep (dict/list answers).
ANSWER_PATTERN = re.compile(r"@([A-Za-z0-9_]+)\s*\[\s*((?:[^\[\]]|\[[^\[\]]*\])*?)\s*\]")


def parse_answers(text: str) -> dict[str, str]:
    """Extract @name[value] pairs from model output, keeping the last value per name."""
    return {name.lower(): value for name, value in ANSWER_PATTERN.findall(text)}


def _normalize_name(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def _as_float(value: str) -> float | None:
    cleaned = value.strip().replace(",", "").replace("$", "").rstrip("%")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _as_literal(value: str):
    try:
        parsed = ast.literal_eval(value.strip())
        return parsed if isinstance(parsed, (dict, list, tuple)) else None
    except (ValueError, SyntaxError):
        return None


def _literals_match(expected, got) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(got, dict)
            and set(expected) == set(got)
            and all(values_match(str(expected[k]), str(got[k])) for k in expected)
        )
    return (
        isinstance(got, (list, tuple))
        and len(expected) == len(got)
        and all(values_match(str(e), str(g)) for e, g in zip(expected, got))
    )


def values_match(expected: str, got: str) -> bool:
    """
    Compare an expected label value with a model value.

    Numeric values match when the model value rounds to the expected one at the
    label's own precision; dict/list values are compared element-wise; other
    values match case-insensitively.
    """
    expected_num = _as_float(expected)
    got_num = _as_float(got)
    if expected_num is not None and got_num is not None:
        if math.isnan(expected_num) or math.isnan(got_num):
            return math.isnan(expected_num) and math.isnan(got_num)
        decimals = len(expected.split(".")[1]) if "." in expected.strip() else 0
        return abs(got_num - expected_num) <= 0.5 * 10 ** -decimals + 1e-9

    expected_lit = _as_literal(expected)
    if expected_lit is not None:
        got_lit = _as_literal(got)
        return got_lit is not None and _literals_match(expected_lit, got_lit)

    return expected.strip().lower() == got.strip().lower()


def score_item(labels: list[tuple[str, str]], answer_text: str) -> dict:
    """
    Score one item: an item is correct only if every labeled sub-answer is matched.

    Names are matched ignoring case, underscores, and hyphens. If the model
    emitted exactly as many answers as expected but under different names,
    they are compared positionally as a fallback.
    """
    parsed = parse_answers(answer_text)
    by_norm = {_normalize_name(name): value for name, value in parsed.items()}

    matched = 0
    for name, expected in labels:
        got = by_norm.get(_normalize_name(name))
        if got is not None and values_match(expected, got):
            matched += 1

    if matched < len(labels) and len(parsed) == len(labels):
        positional = sum(
            values_match(expected, got)
            for (_, expected), got in zip(labels, parsed.values())
        )
        matched = max(matched, positional)

    return {
        "correct": matched == len(labels),
        "sub_answers_total": len(labels),
        "sub_answers_matched": matched,
    }
