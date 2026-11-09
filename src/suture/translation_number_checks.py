"""Conservative English-to-French/Chinese numeral consistency flags.

This module compares explicit numeric mentions. It is not a translation
quality metric, does not verify units or mathematical relations, and has no
universal acceptance threshold. Unsupported dates, versions, ranges, ordinals,
and ambiguous forms are surfaced for review instead of scored as errors.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_DIGIT = r"[0-9０-９]"
_ASCII_BOUNDARY = r"[A-Za-z0-9_]"
_CHINESE_NUMERAL_CHARS = "零〇一二两三四五六七八九十百千万亿点负正"
_CHINESE_DIGIT_VALUES = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}

_EN_SMALL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_EN_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_EN_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000,
              "billion": 1_000_000_000}
_EN_WORDS = set(_EN_SMALL) | set(_EN_TENS) | set(_EN_SCALES) | {"and"}

_FR_SMALL = {
    "zero": 0, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
    "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
    "onze": 11, "douze": 12, "treize": 13, "quatorze": 14,
    "quinze": 15, "seize": 16,
}
_FR_TENS = {"vingt": 20, "trente": 30, "quarante": 40,
            "cinquante": 50, "soixante": 60}
_FR_SCALES = {"cent": 100, "cents": 100, "mille": 1_000,
              "million": 1_000_000, "millions": 1_000_000,
              "milliard": 1_000_000_000, "milliards": 1_000_000_000}
_FR_WORDS = (set(_FR_SMALL) | set(_FR_TENS) | set(_FR_SCALES)
             | {"un", "une", "et"})

_EN_TOKEN_RE = re.compile(
    r"(?i)\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|and)\b"
)
_FR_TOKEN_RE = re.compile(
    r"(?i)\b(?:z[eé]ro|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|"
    r"dix|onze|douze|treize|quatorze|quinze|seize|vingt|trente|quarante|"
    r"cinquante|soixante|cent|cents|mille|million|millions|milliard|"
    r"milliards|et)\b"
)

_EN_RANGE_ATOM = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|and)"
)
_EN_RANGE_NUMBER = rf"{_EN_RANGE_ATOM}(?:[- ]{_EN_RANGE_ATOM})*"
_FR_RANGE_ATOM = (
    r"(?:z[eé]ro|un[e]?|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|"
    r"douze|treize|quatorze|quinze|seize|vingt|trente|quarante|cinquante|"
    r"soixante|cent|cents|mille|million|millions|milliard|milliards|et)"
)
_FR_RANGE_NUMBER = rf"{_FR_RANGE_ATOM}(?:[- ]{_FR_RANGE_ATOM})*"

_UNSUPPORTED_WORD_RE = {
    "en": re.compile(
        r"(?i)\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|"
        r"ninth|tenth|eleventh|twelfth|half|halves|quarter|quarters|"
        r"dozen|dozens|couple)\b"
    ),
    "fr": re.compile(
        r"(?i)\b(?:premier|premi[eè]re|second|seconde|deuxi[eè]me|"
        r"troisi[eè]me|quatri[eè]me|cinqui[eè]me|moiti[eé]|quart|douzaine)\b"
    ),
    "zh": re.compile(r"第[零〇一二两三四五六七八九十百千万亿0-9０-９]+"),
}

_AMBIGUOUS_PATTERNS: Sequence[Tuple[str, re.Pattern[str], Optional[str]]] = (
    ("iso_or_year_first_date", re.compile(r"(?<!\d)\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?!\d)"), None),
    ("slash_date", re.compile(r"(?<!\d)\d{1,2}/\d{1,2}/\d{2,4}(?!\d)"), None),
    ("time", re.compile(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)"), None),
    ("numeric_ordinal", re.compile(r"(?i)(?<!\d)\d+(?:st|nd|rd|th|er|e|ème)(?![A-Za-z])"), None),
    ("version", re.compile(r"(?i)(?<![A-Za-z0-9_])v\d+(?:\.\d+){1,3}(?![A-Za-z0-9_])|(?<!\d)\d+(?:\.\d+){2,}(?!\d)"), None),
    ("numeric_range", re.compile(
        r"(?<![A-Za-z0-9_/])(?:between\s+[+-]?\d+(?:[.,]\d+)?\s+and\s+"
        r"[+-]?\d+(?:[.,]\d+)?|entre\s+[+-]?\d+(?:[.,]\d+)?\s+et\s+"
        r"[+-]?\d+(?:[.,]\d+)?|(?:(?:from|de)\s+)?[+-]?\d+(?:[.,]\d+)?\s*"
        r"(?:--|[-–—]|\bto\b|\bthrough\b|\bà\b|\ba\b|到|至|～|~)\s*"
        r"[+-]?\d+(?:[.,]\d+)?)(?![A-Za-z0-9_])", re.IGNORECASE), None),
    ("written_number_range", re.compile(
        rf"(?i)\b(?:between\s+{_EN_RANGE_NUMBER}\s+and\s+{_EN_RANGE_NUMBER}|"
        rf"from\s+{_EN_RANGE_NUMBER}\s+(?:to|through)\s+{_EN_RANGE_NUMBER}|"
        rf"{_EN_RANGE_NUMBER}\s+(?:to|through)\s+{_EN_RANGE_NUMBER})\b"), "en"),
    ("french_written_number_range", re.compile(
        rf"(?i)\b(?:entre\s+{_FR_RANGE_NUMBER}\s+et\s+{_FR_RANGE_NUMBER}|"
        rf"de\s+{_FR_RANGE_NUMBER}\s+à\s+{_FR_RANGE_NUMBER}|"
        rf"{_FR_RANGE_NUMBER}\s+à\s+{_FR_RANGE_NUMBER})\b"), "fr"),
    ("chinese_date_or_time", re.compile(
        r"[0-9０-９零〇一二两三四五六七八九十百千万亿]+年"
        r"(?:[0-9０-９零〇一二两三四五六七八九十百千万亿]+月"
        r"(?:[0-9０-９零〇一二两三四五六七八九十百千万亿]+[日号])?)?"
        r"|[0-9０-９零〇一二两三四五六七八九十百千万亿]+月"
        r"[0-9０-９零〇一二两三四五六七八九十百千万亿]+[日号]"
        r"|[0-9０-９零〇一二两三四五六七八九十百千万亿]+点"
        r"[0-9０-９零〇一二两三四五六七八九十百千万亿]+分"), None),
    ("chinese_written_range", re.compile(
        r"[零〇一二两三四五六七八九十百千万亿]+\s*(?:到|至|～|~)\s*"
        r"[零〇一二两三四五六七八九十百千万亿]+"), None),
    ("alphanumeric_identifier", re.compile(
        r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_.-]*\d+[A-Za-z0-9_.-]*(?![A-Za-z0-9_])"), None),
)

_CURRENCY_SYMBOLS = "$€£¥￥"


def _language_code(value: str) -> str:
    value = str(value).strip().lower().replace("_", "-")
    if value in {"fr", "french", "français", "francais"} or value.startswith("fr-"):
        return "fr"
    if value in {"zh", "chinese", "中文", "普通话"} or value.startswith("zh-"):
        return "zh"
    raise ValueError("target_language must be French ('fr') or Chinese ('zh')")


def _rational_string(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _ascii_numeric_text(value: str) -> str:
    table = {ord("，"): ",", ord("．"): ".", ord("／"): "/",
             ord("＋"): "+", ord("－"): "-", ord("−"): "-"}
    return unicodedata.normalize("NFKC", value).translate(table)


def _parse_digit_value(raw: str, language: str) -> Optional[Fraction]:
    normalized = _ascii_numeric_text(raw).strip()
    sign = -1 if normalized.startswith("-") else 1
    normalized = normalized.lstrip("+-")
    if "/" in normalized:
        pieces = normalized.split("/")
        if len(pieces) != 2:
            return None
        numerator = re.sub(r"[\s\u00a0\u202f,]", "", pieces[0])
        denominator = re.sub(r"[\s\u00a0\u202f,]", "", pieces[1])
        if not numerator.isdigit() or not denominator.isdigit() or int(denominator) == 0:
            return None
        return sign * Fraction(int(numerator), int(denominator))

    normalized = normalized.replace("\u00a0", " ").replace("\u202f", " ")
    if language == "fr":
        normalized = normalized.replace(" ", "").replace(".", "").replace(",", ".")
    else:
        normalized = normalized.replace(" ", "").replace(",", "")
    try:
        return sign * Fraction(Decimal(normalized))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def _number_body_pattern(language: str) -> re.Pattern[str]:
    digit = _DIGIT
    comma = r"[,，]"
    dot = r"[.．]"
    space = r"[ \u00a0\u202f]"
    grouped_space = rf"{digit}{{1,3}}(?:{space}{digit}{{3}})+"
    if language == "fr":
        grouped_dot = rf"{digit}{{1,3}}(?:{dot}{digit}{{3}})+"
        integer = rf"(?:{grouped_space}|{grouped_dot}|{digit}+)"
        fraction = rf"{digit}+{space}*/{space}*{digit}+"
        body = rf"[+−-]?(?:{fraction}|{integer}(?:{comma}{digit}+)?)"
    else:
        grouped_comma = rf"{digit}{{1,3}}(?:{comma}{digit}{{3}})+"
        integer = rf"(?:{grouped_space}|{grouped_comma}|{digit}+)"
        fraction = rf"{digit}+{space}*/{space}*{digit}+"
        body = rf"[+−-]?(?:{fraction}|{integer}(?:{dot}{digit}+)?)"
    return re.compile(rf"(?<![A-Za-z0-9_])(?P<number>{body})(?![A-Za-z0-9_])")


def _localized_separator_ambiguities(text: str, language: str) -> List[Dict[str, Any]]:
    patterns: List[Tuple[str, re.Pattern[str]]] = []
    if language in {"en", "zh"}:
        patterns.append(("comma_may_be_decimal_or_list", re.compile(r"(?<!\d)\d+[,，]\d{1,2}(?!\d)")))
    elif language == "fr":
        patterns.append(("dot_may_be_decimal_or_grouping", re.compile(r"(?<!\d)\d+[.．]\d{1,2}(?!\d)")))
    found: List[Dict[str, Any]] = []
    for reason, pattern in patterns:
        for match in pattern.finditer(text):
            found.append({"span": match.group(0), "start": match.start(),
                          "end": match.end(), "reason": reason})
    return found


def _find_ambiguous_spans(text: str, language: str) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    for reason, pattern, language_filter in _AMBIGUOUS_PATTERNS:
        if language_filter is not None and language_filter != language:
            continue
        for match in pattern.finditer(text):
            found.append({"span": match.group(0), "start": match.start(),
                          "end": match.end(), "reason": reason})
    found.extend(_localized_separator_ambiguities(text, language))
    found.sort(key=lambda item: (item["start"], item["end"]))
    merged: List[Dict[str, Any]] = []
    for item in found:
        if merged and item["start"] < merged[-1]["end"]:
            previous = merged[-1]
            previous["end"] = max(previous["end"], item["end"])
            previous["span"] = text[previous["start"]:previous["end"]]
            reasons = set(previous["reasons"])
            reasons.add(item["reason"])
            previous["reasons"] = sorted(reasons)
        else:
            merged.append({"span": item["span"], "start": item["start"],
                           "end": item["end"], "reasons": [item["reason"]]})
    return merged


def _overlaps(start: int, end: int, spans: Sequence[Any]) -> bool:
    for span in spans:
        if isinstance(span, Mapping):
            span_start, span_end = int(span["start"]), int(span["end"])
        else:
            span_start, span_end = int(span[0]), int(span[1])
        if start < span_end and end > span_start:
            return True
    return False


def _number_marker(text: str, start: int, end: int) -> Tuple[int, int, str, Optional[str]]:
    """Return mention bounds, style, and a literal symbol annotation."""
    left, right = start, end
    prefix = text[max(0, start - 4):start]
    prefix_match = re.search(r"([$€£¥￥])\s*$", prefix)
    suffix = text[end:min(len(text), end + 16)]
    suffix_match = re.match(r"\s*([$€£¥￥])", suffix)
    marker: Optional[str] = None
    style = "number"
    if prefix_match:
        left = start - len(prefix_match.group(0))
        marker = prefix_match.group(1)
        style = "currency"
    if suffix_match:
        right = end + len(suffix_match.group(0))
        marker = suffix_match.group(1)
        style = "currency"
    prefix_percent = re.search(r"百分之\s*$", text[max(0, start - 5):start])
    suffix_percent = re.match(r"\s*(?:%|％)", suffix)
    if suffix_percent is None:
        suffix_percent = re.match(
            r"\s*(?:percent(?:age)?s?|per\s+cent|pour\s+cent)\b", suffix, re.IGNORECASE
        )
    if prefix_percent:
        left = start - len(prefix_percent.group(0))
        style = "percent"
        marker = "百分之"
    elif suffix_percent:
        right = max(right, end + len(suffix_percent.group(0)))
        style = "percent"
        marker = suffix_percent.group(0).strip()
    elif re.match(r"\s*[%％]", suffix):
        match = re.match(r"\s*[%％]", suffix)
        assert match is not None
        right = max(right, end + len(match.group(0)))
        style = "percent"
        marker = match.group(0).strip()
    return left, right, style, marker


def _extract_digit_mentions(text: str, language: str,
                            ambiguous: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    mentions: List[Dict[str, Any]] = []
    pattern = _number_body_pattern(language)
    for match in pattern.finditer(text):
        start, end = match.span("number")
        if _overlaps(start, end, ambiguous):
            continue
        value = _parse_digit_value(match.group("number"), language)
        if value is None:
            continue
        left, right, style, marker = _number_marker(text, start, end)
        mentions.append({
            "span": text[left:right],
            "start": left,
            "end": right,
            "normalized_value": _rational_string(value),
            "kind": "fraction" if "/" in match.group("number") else style,
            "marker": marker,
            "source_form": "digits",
        })
    return mentions


def _parse_english_words(words: Sequence[str]) -> Optional[int]:
    total = 0
    current = 0
    previous = ""
    if not words:
        return None
    cleaned: List[str] = []
    for index, word in enumerate(words):
        if word == "and":
            if index == 0 or index == len(words) - 1 or words[index - 1] not in _EN_SCALES:
                return None
            continue
        cleaned.append(word)
    for word in cleaned:
        if word in _EN_SMALL:
            value = _EN_SMALL[word]
            if current == 0:
                current = value
            elif current in _EN_TENS.values() and 0 < value < 10:
                current += value
            elif current == 10 and 0 < value < 10:
                current += value
            elif current >= 100 and 0 < value < 10:
                current += value
            else:
                return None
        elif word in _EN_TENS:
            if current == 0:
                current = _EN_TENS[word]
            elif current >= 100:
                current += _EN_TENS[word]
            else:
                return None
        elif word in _EN_SCALES:
            scale = _EN_SCALES[word]
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0
        else:
            return None
        previous = word
    _ = previous
    return total + current


def _fold_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.lower())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _parse_french_words(raw_words: Sequence[str]) -> Optional[int]:
    words = [_fold_accents(word).replace("’", "'") for word in raw_words]
    if not words:
        return None
    if words[0] in {"un", "une"}:
        # Do not guess whether a stand-alone French un/une is a numeral or article.
        # Accept only explicit constructions such as "un seul" or "un million".
        if len(words) == 2 and words[1] in {"seul", "seule"}:
            return 1
        if len(words) < 2 or words[1] not in _FR_SCALES:
            return None
    total = 0
    current = 0
    i = 0
    while i < len(words):
        word = words[i]
        if word == "et":
            if i + 1 >= len(words) or words[i + 1] not in {"un", "une", "onze"}:
                return None
            if current < 20:
                return None
            if words[i + 1] == "onze":
                if current != 60:
                    return None
                current += 11
            else:
                current += 1
            i += 2
            continue
        if word == "quatre" and i + 1 < len(words) and words[i + 1] == "vingt":
            if current != 0:
                return None
            current = 80
            i += 2
            if i < len(words) and words[i] in {"dix", "onze", "douze", "treize", "quatorze", "quinze", "seize"}:
                current += _FR_SMALL.get(words[i], 10)
                i += 1
            continue
        if word in _FR_SMALL or word in {"un", "une"}:
            value = _FR_SMALL.get(word, 1)
            if current == 0:
                current = value
            elif current in _FR_TENS.values() and 0 < value < 10:
                current += value
            elif current == 10 and 0 < value < 10:
                current += value
            elif current in {60, 80} and 10 <= value <= 16:
                current += value
            elif current in {70, 80, 90} and 0 < value < 10:
                current += value
            elif current >= 100 and 0 < value < 10:
                current += value
            else:
                return None
        elif word in _FR_TENS:
            if current == 0:
                current = _FR_TENS[word]
            elif current >= 100:
                current += _FR_TENS[word]
            else:
                return None
        elif word in _FR_SCALES:
            scale = _FR_SCALES[word]
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0
        elif word == "zero":
            if current != 0 or total != 0:
                return None
        else:
            return None
        i += 1
    return total + current


def _parse_chinese_words(raw: str) -> Optional[Fraction]:
    text = raw
    sign = 1
    if text.startswith(("负", "－", "-")):
        sign = -1
        text = text[1:]
    elif text.startswith(("正", "+")):
        text = text[1:]
    if not text:
        return None
    if "点" in text:
        if text.count("点") != 1:
            return None
        left, right = text.split("点", 1)
        if not left or not right or any(ch not in _CHINESE_DIGIT_VALUES for ch in right):
            return None
        integer = _parse_chinese_integer(left)
        if integer is None:
            return None
        decimal_digits = "".join(str(_CHINESE_DIGIT_VALUES[ch]) for ch in right)
        return sign * Fraction(Decimal(f"{integer}.{decimal_digits}"))
    integer = _parse_chinese_integer(text)
    return None if integer is None else sign * Fraction(integer)


def _parse_chinese_integer(text: str) -> Optional[int]:
    if not text or any(ch not in _CHINESE_NUMERAL_CHARS for ch in text):
        return None
    if all(ch in _CHINESE_DIGIT_VALUES for ch in text):
        return int("".join(str(_CHINESE_DIGIT_VALUES[ch]) for ch in text))
    total = 0
    section = 0
    current: Optional[int] = None
    unit_values = {"十": 10, "百": 100, "千": 1_000}
    for char in text:
        if char in _CHINESE_DIGIT_VALUES:
            if current is not None and current != 0:
                return None
            current = _CHINESE_DIGIT_VALUES[char]
        elif char in unit_values:
            unit = unit_values[char]
            value = 1 if current is None and unit == 10 else current
            if value is None:
                return None
            section += value * unit
            current = None
        elif char == "万":
            section += current or 0
            total += (section or 1) * 10_000
            section = 0
            current = None
        elif char == "亿":
            section += current or 0
            total += (section or 1) * 100_000_000
            section = 0
            current = None
        else:
            return None
    return total + section + (current or 0)


def _word_mentions(text: str, language: str,
                    ambiguous: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    mentions: List[Dict[str, Any]] = []
    ambiguous_articles: List[Dict[str, Any]] = []
    unhandled: List[Dict[str, Any]] = []
    unsupported_re = _UNSUPPORTED_WORD_RE[language]
    for match in unsupported_re.finditer(text):
        if not _overlaps(match.start(), match.end(), ambiguous):
            unhandled.append({"span": match.group(0), "start": match.start(),
                              "end": match.end(), "reason": "unsupported_ordinal_or_fraction_word"})

    if language == "zh":
        # Percentages written as 百分之三十 are one mention, not two.
        percent_pattern = re.compile(r"百分之[负正]?[零〇一二两三四五六七八九十百千万亿点]+")
        covered: List[Tuple[int, int]] = []
        for match in percent_pattern.finditer(text):
            if _overlaps(match.start(), match.end(), ambiguous):
                continue
            value = _parse_chinese_words(match.group(0)[3:])
            if value is None:
                unhandled.append({"span": match.group(0), "start": match.start(),
                                  "end": match.end(), "reason": "unsupported_chinese_percent"})
                continue
            mentions.append({"span": match.group(0), "start": match.start(),
                             "end": match.end(), "normalized_value": _rational_string(value),
                             "kind": "percent", "marker": "百分之", "source_form": "number_word"})
            covered.append(match.span())
        numeral_re = re.compile(rf"[{re.escape(_CHINESE_NUMERAL_CHARS)}]+")
        for match in numeral_re.finditer(text):
            if _overlaps(match.start(), match.end(), ambiguous) or _overlaps(match.start(), match.end(), [
                {"start": start, "end": end} for start, end in covered
            ]):
                continue
            value = _parse_chinese_words(match.group(0))
            if value is None:
                unhandled.append({"span": match.group(0), "start": match.start(),
                                  "end": match.end(), "reason": "unsupported_chinese_number"})
                continue
            mentions.append({"span": match.group(0), "start": match.start(),
                             "end": match.end(), "normalized_value": _rational_string(value),
                             "kind": "number_word", "marker": None, "source_form": "number_word"})
        mentions.sort(key=lambda item: (item["start"], item["end"]))
        return mentions, ambiguous_articles, unhandled

    token_re = _EN_TOKEN_RE if language == "en" else _FR_TOKEN_RE
    article_re = re.compile(r"(?i)\b(?:a|an)\b") if language == "en" else re.compile(r"(?i)\b(?:un|une)\b")
    parsed_article_spans: List[Tuple[int, int]] = []
    if language == "fr":
        # Treat "un seul" / "une seule" as explicit one, but not ordinary articles.
        for match in re.finditer(r"(?i)\b(un|une)\s+(seul|seule)\b", text):
            if _overlaps(match.start(), match.end(), ambiguous):
                continue
            mentions.append({"span": match.group(0), "start": match.start(), "end": match.end(),
                             "normalized_value": "1", "kind": "number_word", "marker": None,
                             "source_form": "number_word"})
            parsed_article_spans.append(match.span())
    tokens = list(token_re.finditer(text))
    scale_words = _EN_SCALES if language == "en" else _FR_SCALES
    small_words = _EN_SMALL if language == "en" else _FR_SMALL
    tens_words = _EN_TENS if language == "en" else _FR_TENS
    candidate_word_set = _EN_WORDS if language == "en" else _FR_WORDS

    i = 0
    while i < len(tokens):
        first = tokens[i]
        first_word = first.group(0).lower()
        if first_word not in candidate_word_set or first_word == "and" or first_word == "et":
            i += 1
            continue
        run = [first_word]
        j = i + 1
        end = first.end()
        while j < len(tokens):
            gap = text[end:tokens[j].start()]
            next_word = tokens[j].group(0).lower()
            if not re.fullmatch(r"[\s-]+", gap) or next_word not in candidate_word_set:
                break
            if language == "en" and next_word == "and" and run[-1] not in _EN_SCALES:
                break
            if language == "fr" and next_word == "et" and run[-1] not in _FR_TENS:
                # French conjunction in a numeral is limited to tens plus un/une,
                # or soixante et onze.
                following = tokens[j + 1].group(0).lower() if j + 1 < len(tokens) else ""
                if following not in {"un", "une", "onze"} or (following == "onze" and run[-1] != "soixante"):
                    break
            run.append(next_word)
            end = tokens[j].end()
            j += 1
        start = first.start()
        if _overlaps(start, end, ambiguous) or _overlaps(start, end, parsed_article_spans):
            i = max(i + 1, j)
            continue
        if language == "en":
            value = _parse_english_words(run)
        else:
            value = _parse_french_words(run)
        if value is not None:
            left, right, style, marker = _number_marker(text, start, end)
            mentions.append({"span": text[left:right], "start": left, "end": right,
                             "normalized_value": str(value),
                             "kind": style if style != "number" else "number_word",
                             "marker": marker, "source_form": "number_word"})
            parsed_article_spans.extend((token.start(), token.end()) for token in tokens[i:j]
                                        if token.group(0).lower() in {"un", "une"})
            i = max(i + 1, j)
        else:
            i += 1

    for match in article_re.finditer(text):
        if not _overlaps(match.start(), match.end(), ambiguous) and not _overlaps(
            match.start(), match.end(), parsed_article_spans
        ):
            ambiguous_articles.append({"span": match.group(0), "start": match.start(),
                                       "end": match.end(), "reason": "article_not_counted_as_one"})
    return mentions, ambiguous_articles, unhandled


def _multiset(mentions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    counts = Counter(str(item["normalized_value"]) for item in mentions)
    return [{"normalized_value": value, "count": counts[value]} for value in sorted(counts)]


def _multiset_delta(source: Sequence[Mapping[str, Any]], target: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    source_counts = Counter(str(item["normalized_value"]) for item in source)
    target_counts = Counter(str(item["normalized_value"]) for item in target)
    missing = [{"normalized_value": value, "count": count}
               for value, count in sorted((source_counts - target_counts).items())]
    extra = [{"normalized_value": value, "count": count}
             for value, count in sorted((target_counts - source_counts).items())]
    return missing, extra


def _compare(source: Sequence[Mapping[str, Any]], target: Sequence[Mapping[str, Any]],
             ambiguity: Sequence[Mapping[str, Any]], article_candidates: Sequence[Mapping[str, Any]],
             unhandled: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    missing, extra = _multiset_delta(source, target)
    if not missing and not extra:
        status = "review-needed" if ambiguity or unhandled else "match"
    elif ambiguity or unhandled or article_candidates:
        status = "review-needed"
    else:
        status = "mismatch"
    return {
        "status": status,
        "source_multiset": _multiset(source),
        "target_multiset": _multiset(target),
        "missing_from_target": missing,
        "extra_in_target": extra,
    }


def _unique_spans(*groups: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output: List[Dict[str, Any]] = []
    for group in groups:
        for item in group:
            key = (int(item["start"]), int(item["end"]), str(item["span"]))
            if key not in seen:
                seen.add(key)
                output.append(dict(item))
    return sorted(output, key=lambda item: (item["start"], item["end"]))


def _audit_text(text: str, language: str) -> Dict[str, Any]:
    ambiguous = _find_ambiguous_spans(text, language)
    digits = _extract_digit_mentions(text, language, ambiguous)
    words, article_candidates, unhandled_words = _word_mentions(text, language, ambiguous)
    unhandled = _unique_spans(unhandled_words)
    ambiguous_all = _unique_spans(ambiguous)
    signals = _unique_spans(
        [{"span": item["span"], "start": item["start"], "end": item["end"], "reason": "digits"} for item in digits],
        [{"span": item["span"], "start": item["start"], "end": item["end"], "reason": "number_word"} for item in words],
        ambiguous_all,
        unhandled,
    )
    return {
        "digit_mentions": digits,
        "supported_number_words": words,
        "ambiguous_spans": ambiguous_all,
        "unhandled_spans": unhandled,
        "article_candidates_not_counted": article_candidates,
        "has_numeric_signal": bool(signals),
        "numeric_signal_count": len(signals),
    }


def audit_numbers(source_text: str, target_text: str, target_language: str) -> Dict[str, Any]:
    """Flag differences in explicit numeric mentions in an EN to FR/ZH pair.

    ``status`` is ``match``, ``mismatch``, or ``review-needed`` for the broad
    comparison. It is a consistency flag only, never a confirmed translation
    error. The strict digit comparison is reported separately from supported
    number-word normalization. French ``un/une`` and English ``a/an`` are not
    counted as one; if a numeric difference could be explained by such an
    ambiguous article, the broad result is ``review-needed``.
    """
    if not isinstance(source_text, str) or not isinstance(target_text, str):
        raise TypeError("source_text and target_text must be strings")
    language = _language_code(target_language)
    source = _audit_text(source_text, "en")
    target = _audit_text(target_text, language)

    source_digits = source["digit_mentions"]
    target_digits = target["digit_mentions"]
    source_all = source_digits + source["supported_number_words"]
    target_all = target_digits + target["supported_number_words"]
    strict = _compare(
        source_digits, target_digits,
        source["ambiguous_spans"] + target["ambiguous_spans"],
        source["article_candidates_not_counted"] + target["article_candidates_not_counted"],
        source["unhandled_spans"] + target["unhandled_spans"],
    )
    broad = _compare(
        source_all, target_all,
        source["ambiguous_spans"] + target["ambiguous_spans"],
        source["article_candidates_not_counted"] + target["article_candidates_not_counted"],
        source["unhandled_spans"] + target["unhandled_spans"],
    )
    caveats = [
        "These are numeral-consistency flags, not confirmed translation errors or an error-rate estimate.",
        "No units, currency meaning, dates, mathematical relations, or semantic correctness are verified.",
        "Dates, times, versions, ranges, unsupported number words, and ambiguous locale separators require review.",
        "English a/an and French un/une are not automatically counted as one; they are reported as article candidates.",
        "No universal acceptance threshold is defined. Review-needed records must remain visible in any aggregate.",
    ]
    all_ambiguous = source["ambiguous_spans"] + target["ambiguous_spans"]
    all_unhandled = source["unhandled_spans"] + target["unhandled_spans"]
    status = broad["status"]
    return {
        "schema_version": 1,
        "check_kind": "numeral_consistency_flag",
        "source_language": "en",
        "target_language": language,
        "status": status,
        "strict_digit_comparison": strict,
        "supported_number_comparison": broad,
        "source": source,
        "target": target,
        "unhandled_source_spans": source["unhandled_spans"],
        "unhandled_target_spans": target["unhandled_spans"],
        "ambiguous_source_spans": source["ambiguous_spans"],
        "ambiguous_target_spans": target["ambiguous_spans"],
        "source_has_numeric_signal": source["has_numeric_signal"],
        "target_has_numeric_signal": target["has_numeric_signal"],
        "source_numeric_signal_count": source["numeric_signal_count"],
        "target_numeric_signal_count": target["numeric_signal_count"],
        "caveats": caveats,
    }


__all__ = ["audit_numbers"]
