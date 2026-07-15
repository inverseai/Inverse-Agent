"""Redaction and egress helpers."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from unicodedata import category, normalize

__all__ = [
    "SOURCE_INSTRUCTION_REDACTION_MARKER",
    "RedactionResult",
    "SecretSpan",
    "SourceInstructionResult",
    "neutralize_source_instructions",
    "private_key_spans",
    "redact_text",
    "secret_spans",
]

_PEM_TOKEN = re.compile(
    r"(?=(?P<token>-----(?P<kind>BEGIN|END) |-----))",
    re.IGNORECASE | re.ASCII,
)
_PRIVATE_KEY_LABEL = re.compile(r"PRIVATE KEY", re.IGNORECASE | re.ASCII)
_PEM_MARKER_CLOSE_LENGTH = 5

_SOURCE_LABEL_AUTHORITY = (
    r"(?:reviewer|assistant|model|system|developer|user|prompt|instruction)"
    r"(?=\s*[:\-])"
)
_SOURCE_PHRASE_AUTHORITY = (
    r"(?:"
    r"(?:system|developer|reviewer|assistant|model|user)\s+"
    r"(?:message|prompt|instructions?)|"
    r"language\s+model|ai\s+(?:assistant|model)"
    r")"
)
_SOURCE_AUTHORITY = rf"(?:{_SOURCE_LABEL_AUTHORITY}|{_SOURCE_PHRASE_AUTHORITY})"
_SOURCE_EXPLICIT_AUTHORITY = rf"(?:(?:reviewer|assistant)(?=\s*[:\-])|{_SOURCE_PHRASE_AUTHORITY})"
_SOURCE_OVERRIDE = r"(?:ignore|disregard|override)"


def _compile_source_context_patterns(
    authority: str, *, allow_assignment: bool
) -> tuple[re.Pattern[str], ...]:
    assignment_guard = "" if allow_assignment else r"(?!\s*[:=])"
    return (
        re.compile(
            rf"(?i)\b{authority}\b.{{0,160}}"
            rf"\b{_SOURCE_OVERRIDE}\b{assignment_guard}"
        ),
        re.compile(
            rf"(?i)\b{_SOURCE_OVERRIDE}\b{assignment_guard}.{{0,160}}"
            rf"\b{authority}\b"
        ),
        # Outcome manipulation is review-specific. Bare ``complete`` is not an
        # outcome here because ordinary code commonly returns that value.
        re.compile(
            rf"(?i)\b{authority}\b.{{0,160}}"
            rf"\b(?:return|output|respond(?:\s+with)?|report)\b{assignment_guard}.{{0,80}}"
            r"\b(?:pass|no\s+findings?|verdict|investigation\s+complete)\b"
        ),
        re.compile(
            rf"(?i)\b{authority}\b.{{0,160}}\bmark\b{assignment_guard}.{{0,80}}"
            r"\b(?:pass|complete|no\s+findings?|verdict)\b"
        ),
    )


_SOURCE_EXECUTABLE_INSTRUCTION_PATTERN = re.compile(
    r"(?i)(?P<reviewer>\breviewer\b)\s*=\s*(?P<model>\bmodel\b).{0,160}"
    r"(?P<return>\breturn\b).{0,80}(?P<finding>\bfinding\b)"
)
_SOURCE_EXECUTABLE_GROUPS = ("reviewer", "model", "return", "finding")
_SOURCE_CONTEXT_INSTRUCTION_PATTERNS = (
    # Explicit authority claims paired with an instruction override. Keeping
    # the authority vocabulary phrase-based avoids treating ordinary code such
    # as ``return self.model.forward(x)`` as a prompt-injection payload.
    *_compile_source_context_patterns(_SOURCE_EXPLICIT_AUTHORITY, allow_assignment=True),
    *_compile_source_context_patterns(_SOURCE_AUTHORITY, allow_assignment=False),
    # Preserve the existing fail-closed treatment of an instruction disguised
    # as executable reviewer/model plumbing without broad bare-word matching.
    _SOURCE_EXECUTABLE_INSTRUCTION_PATTERN,
)
_SOURCE_STRONG_CONTEXT_INSTRUCTION_PATTERNS = _compile_source_context_patterns(
    _SOURCE_PHRASE_AUTHORITY,
    allow_assignment=True,
)
_SOURCE_STANDALONE_INSTRUCTION_PATTERN = re.compile(
    r"(?i)^(?:please\s+)?(?:ignore|disregard|override)\b(?!\s*[:=]).{0,80}"
    r"\b(?:all|any|previous|prior|above)\b.{0,80}"
    r"\b(?:instructions?|system\s+(?:message|prompt)|developer\s+message)\b"
)
_SOURCE_STANDALONE_FRAGMENT_PATTERN = re.compile(
    r"(?i)^(?:(?:please\s+)?(?:ignore|disregard|override)\b(?!\s*[:=])"
    r"(?:\s+(?:all|any|previous|prior|above))?\s*[.!:]?$|"
    r"(?:(?:all|any|previous|prior|above)\s+)?"
    r"(?:instructions?|system\s+(?:message|prompt)|developer\s+message)\b"
    r")"
)
_SOURCE_COMMENT_MARKERS = ("//", "#", "/*", "<!--", "-- ")
_SOURCE_RUST_RAW_STRING_OPEN = re.compile(r'(?:br|r)(?P<hashes>#+)"')
_SOURCE_CPP_RAW_STRING_OPEN = re.compile(
    r'(?:(?:u8|u|U|L)?R)"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\('
)
_SOURCE_CSHARP_VERBATIM_OPEN = re.compile(r'(?:\$@|@\$|@)"')
_SOURCE_REVIEW_LABEL_LINE = re.compile(r"(?i)^(?:reviewer|assistant)\s*[:\-]\s*$")
_SOURCE_LABEL_AUTHORITY_PATTERN = re.compile(rf"(?i)\b{_SOURCE_LABEL_AUTHORITY}\b")
_SOURCE_AUTHORITY_PATTERN = re.compile(rf"(?i)\b{_SOURCE_AUTHORITY}\b")
_SOURCE_CONTEXT_PAYLOAD_FRAGMENT_PATTERN = re.compile(
    r"(?i)(?:"
    rf"\b{_SOURCE_OVERRIDE}\b(?!\s*[:=]).{{0,80}}\b(?:instructions?|findings?)\b|"
    r"\b(?:return|output|respond|report|mark)\b(?!\s*[:=]).{0,80}"
    r"\b(?:pass|no\s+findings?|verdict|complete|finding)\b"
    r")"
)
_SOURCE_COMMENT_PREFIX_PATTERN = re.compile(r"^(?:(?:<!--|-->|/\*|\*/|#+|/{2,}|\*+|--\s+)\s*)+")
_SOURCE_COMMENT_SUFFIX_PATTERN = re.compile(r"\s*(?:\*/|-->)\s*$")
SOURCE_INSTRUCTION_REDACTION_MARKER = "[untrusted source instruction redacted]"

_SECRET_PATTERNS = (
    (
        "key-value-secret",
        re.compile(
            r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{8,})"
        ),
    ),
    ("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("bearer-token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}")),
    (
        "github-token",
        re.compile(r"\b(?:gh[oprsu]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("provider-token", re.compile(r"\b(?:sk|hf)_[A-Za-z0-9_-]{20,}\b|\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    (
        "credential-url",
        re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^\s:/@]+:[^\s/@]+@"),
    ),
)


@dataclass(frozen=True)
class RedactionResult:
    text: str
    blocked: bool
    matches: tuple[str, ...]


@dataclass(frozen=True)
class SecretSpan:
    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class SourceInstructionResult:
    text: str
    redacted: bool
    incomplete: bool
    redacted_lines: tuple[int, ...] = ()


@dataclass(frozen=True)
class _InstructionCommentSpan:
    start: int
    marker: str
    end: int | None
    closer: str | None


@dataclass(frozen=True)
class _InstructionLexicalLine:
    comment_spans: tuple[_InstructionCommentSpan, ...]
    unquoted_text: str
    member: bool


def _instruction_probe(
    value: str,
    *,
    check: Callable[[], None] | None = None,
) -> str:
    normalized = normalize("NFKC", value)
    output: list[str] = []
    for index, character in enumerate(normalized):
        if check is not None and index % 16384 == 0:
            check()
        if category(character) != "Cf":
            output.append(character)
    return "".join(output)


def _instruction_has_diff_prefix(value: str) -> bool:
    return value[:1] in {"+", "-", " "} and not value.startswith(("-- ", "-->"))


def _instruction_record_content(value: str) -> str:
    # ``-- `` is a raw SQL/Haskell comment marker as well as an ambiguous diff
    # shape. Preserve it: a real removed line still retains its leading ``-``
    # for changed-line accounting after neutralization.
    return value[1:] if _instruction_has_diff_prefix(value) else value


def _instruction_content(value: str) -> str:
    return _instruction_record_content(value).lstrip()


def _instruction_find_unescaped(value: str, needle: str, start: int) -> int | None:
    position = value.find(needle, start)
    while position >= 0:
        backslashes = 0
        cursor = position - 1
        while cursor >= 0 and value[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return position
        position = value.find(needle, position + len(needle))
    return None


def _instruction_find_verbatim_close(value: str, start: int) -> int | None:
    position = value.find('"', start)
    while position >= 0:
        if position + 1 < len(value) and value[position + 1] == '"':
            position = value.find('"', position + 2)
            continue
        return position
    return None


def _instruction_has_line_continuation(value: str) -> bool:
    stripped = value.rstrip()
    backslashes = 0
    for character in reversed(stripped):
        if character != "\\":
            break
        backslashes += 1
    return backslashes % 2 == 1


def _instruction_mask_range(
    output: list[str],
    start: int,
    end: int,
    *,
    check: Callable[[], None] | None = None,
) -> None:
    for index in range(start, end):
        if check is not None and (index - start) % 16384 == 0:
            check()
        output[index] = " "


def _instruction_token_boundary(value: str, index: int) -> bool:
    return index == 0 or not (value[index - 1].isalnum() or value[index - 1] == "_")


def _instruction_lexical_lines(
    values: list[str],
    *,
    check: Callable[[], None] | None = None,
) -> tuple[_InstructionLexicalLine, ...]:
    """Scan comments and mask literals once, carrying multiline lexical state."""

    lines: list[_InstructionLexicalLine] = []
    block_closer: str | None = None
    literal_closer: str | None = None
    literal_escape_aware = False
    literal_doubled_closer = False
    for line_index, value in enumerate(values):
        if check is not None and line_index % 128 == 0:
            check()
        content = _instruction_record_content(value)
        unquoted = list(content)
        member = not content.strip()
        comment_spans: list[_InstructionCommentSpan] = []
        index = 0
        next_character_check = 0

        if block_closer is not None:
            member = True
            comment_start = len(content) - len(content.lstrip())
            close = content.find(block_closer)
            if close < 0:
                comment_spans.append(_InstructionCommentSpan(comment_start, "", None, None))
                lines.append(_InstructionLexicalLine(tuple(comment_spans), content, member))
                continue
            comment_end = close + len(block_closer)
            comment_spans.append(
                _InstructionCommentSpan(
                    comment_start,
                    "",
                    comment_end,
                    block_closer,
                )
            )
            block_closer = None
            index = comment_end

        while index < len(content):
            if check is not None and index >= next_character_check:
                check()
                next_character_check = index + 16384
            if literal_closer is not None:
                if literal_doubled_closer:
                    literal_close = _instruction_find_verbatim_close(content, index)
                elif literal_escape_aware:
                    literal_close = _instruction_find_unescaped(content, literal_closer, index)
                else:
                    raw_close = content.find(literal_closer, index)
                    literal_close = raw_close if raw_close >= 0 else None
                if literal_close is None:
                    _instruction_mask_range(unquoted, index, len(content), check=check)
                    index = len(content)
                    break
                literal_end = literal_close + len(literal_closer)
                _instruction_mask_range(unquoted, index, literal_end, check=check)
                index = literal_end
                literal_closer = None
                literal_escape_aware = False
                literal_doubled_closer = False
                continue

            csharp_verbatim = _SOURCE_CSHARP_VERBATIM_OPEN.match(content, index)
            if csharp_verbatim is not None and _instruction_token_boundary(content, index):
                verbatim_close = _instruction_find_verbatim_close(content, csharp_verbatim.end())
                if verbatim_close is None:
                    _instruction_mask_range(unquoted, index, len(content), check=check)
                    literal_closer = '"'
                    literal_escape_aware = False
                    literal_doubled_closer = True
                    index = len(content)
                    break
                literal_end = verbatim_close + 1
                _instruction_mask_range(unquoted, index, literal_end, check=check)
                index = literal_end
                continue

            rust_raw = _SOURCE_RUST_RAW_STRING_OPEN.match(content, index)
            if rust_raw is not None and _instruction_token_boundary(content, index):
                closer = f'"{rust_raw.group("hashes")}'
                close = content.find(closer, rust_raw.end())
                if close < 0:
                    _instruction_mask_range(unquoted, index, len(content), check=check)
                    literal_closer = closer
                    literal_escape_aware = False
                    literal_doubled_closer = False
                    index = len(content)
                    break
                literal_end = close + len(closer)
                _instruction_mask_range(unquoted, index, literal_end, check=check)
                index = literal_end
                continue

            cpp_raw = _SOURCE_CPP_RAW_STRING_OPEN.match(content, index)
            if cpp_raw is not None and _instruction_token_boundary(content, index):
                closer = f'){cpp_raw.group("delimiter")}"'
                close = content.find(closer, cpp_raw.end())
                if close < 0:
                    _instruction_mask_range(unquoted, index, len(content), check=check)
                    literal_closer = closer
                    literal_escape_aware = False
                    literal_doubled_closer = False
                    index = len(content)
                    break
                literal_end = close + len(closer)
                _instruction_mask_range(unquoted, index, literal_end, check=check)
                index = literal_end
                continue

            triple = next(
                (marker for marker in ('"""', "'''") if content.startswith(marker, index)),
                None,
            )
            if triple is not None:
                triple_close = _instruction_find_unescaped(content, triple, index + len(triple))
                if triple_close is None:
                    _instruction_mask_range(unquoted, index, len(content), check=check)
                    literal_closer = triple
                    literal_escape_aware = True
                    literal_doubled_closer = False
                    index = len(content)
                    break
                literal_end = triple_close + len(triple)
                _instruction_mask_range(unquoted, index, literal_end, check=check)
                index = literal_end
                continue

            character = content[index]
            if character in {'"', "'", "`"}:
                quote_close = _instruction_find_unescaped(content, character, index + 1)
                if quote_close is None:
                    # Unmatched apostrophes are valid Rust lifetimes and Haskell
                    # identifier primes. Other unmatched delimiters conservatively
                    # mask the rest of this logical line.
                    continued = _instruction_has_line_continuation(content)
                    if character == "'" and not continued:
                        index += 1
                        continue
                    _instruction_mask_range(unquoted, index, len(content), check=check)
                    if character == "`" or continued:
                        literal_closer = character
                        literal_escape_aware = True
                        literal_doubled_closer = False
                    index = len(content)
                    break
                _instruction_mask_range(unquoted, index, quote_close + 1, check=check)
                index = quote_close + 1
                continue

            marker_match: str | None = None
            for marker in _SOURCE_COMMENT_MARKERS:
                if not content.startswith(marker, index):
                    continue
                if marker == "//" and index > 0 and content[index - 1] == ":":
                    continue
                marker_match = marker
                break
            if marker_match is not None:
                member = True
                if marker_match in {"/*", "<!--"}:
                    closer = "*/" if marker_match == "/*" else "-->"
                    close = content.find(closer, index + len(marker_match))
                    if close < 0:
                        comment_spans.append(
                            _InstructionCommentSpan(index, marker_match, None, None)
                        )
                        block_closer = closer
                        break
                    else:
                        comment_end = close + len(closer)
                        comment_spans.append(
                            _InstructionCommentSpan(
                                index,
                                marker_match,
                                comment_end,
                                closer,
                            )
                        )
                        index = comment_end
                        continue
                comment_spans.append(_InstructionCommentSpan(index, marker_match, None, None))
                break
            index += 1

        lines.append(
            _InstructionLexicalLine(
                comment_spans=tuple(comment_spans),
                unquoted_text="".join(unquoted),
                member=member,
            )
        )
    return tuple(lines)


def _instruction_is_review_label(value: str) -> bool:
    return _SOURCE_REVIEW_LABEL_LINE.fullmatch(_instruction_content(value)) is not None


def _instruction_has_label_authority(value: str) -> bool:
    return _SOURCE_LABEL_AUTHORITY_PATTERN.search(value) is not None


def _instruction_has_authority(value: str) -> bool:
    return _SOURCE_AUTHORITY_PATTERN.search(value) is not None


def _instruction_directive_window(values: tuple[str, ...]) -> str:
    """Return normalized comment prose for an anchored split directive check."""

    parts: list[str] = []
    for value in values:
        content = _instruction_content(value)
        content = _SOURCE_COMMENT_PREFIX_PATTERN.sub("", content)
        content = _SOURCE_COMMENT_SUFFIX_PATTERN.sub("", content)
        stripped = content.strip()
        if stripped:
            parts.append(stripped)
    return " ".join(parts)


def _instruction_has_standalone_match(
    pattern: re.Pattern[str],
    values: tuple[str, ...],
    *,
    check: Callable[[], None] | None = None,
) -> bool:
    """Match an anchored directive directly or after any colon-delimited label."""

    if check is not None:
        check()
    window = _instruction_directive_window(values)
    # Both standalone patterns have fixed, bounded tails. Limiting each candidate
    # prevents a colon-heavy source line from turning label checks quadratic.
    candidate_limit = 512
    if pattern.search(window[:candidate_limit]) is not None:
        return True

    awaiting_label_payload = False
    for index, character in enumerate(window):
        if check is not None and index % 16384 == 0:
            check()
        if character == ":":
            awaiting_label_payload = True
            continue
        if not awaiting_label_payload or character.isspace():
            continue
        if pattern.search(window[index : index + candidate_limit]) is not None:
            return True
        awaiting_label_payload = False
    return False


def _instruction_has_standalone_instruction(
    values: tuple[str, ...],
    *,
    check: Callable[[], None] | None = None,
) -> bool:
    return _instruction_has_standalone_match(
        _SOURCE_STANDALONE_INSTRUCTION_PATTERN,
        values,
        check=check,
    )


def _instruction_has_standalone_fragment(
    values: tuple[str, ...],
    *,
    check: Callable[[], None] | None = None,
) -> bool:
    return _instruction_has_standalone_match(
        _SOURCE_STANDALONE_FRAGMENT_PATTERN,
        values,
        check=check,
    )


def _instruction_executable_text(
    lexical: _InstructionLexicalLine,
    *,
    check: Callable[[], None] | None = None,
) -> str:
    """Return quote-masked source with every recognized comment span masked."""

    output = list(lexical.unquoted_text)
    for span in lexical.comment_spans:
        _instruction_mask_range(
            output,
            span.start,
            span.end or len(output),
            check=check,
        )
    return "".join(output)


def _instruction_executable_match_indexes(
    probes: tuple[str, ...],
    indexes: tuple[int, ...],
) -> tuple[int, ...]:
    """Map exact executable-pattern capture groups back to contributing lines."""

    parts = tuple(probes[index] for index in indexes)
    match = _SOURCE_EXECUTABLE_INSTRUCTION_PATTERN.search(" ".join(parts))
    if match is None:
        return ()
    ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for index, part in zip(indexes, parts, strict=True):
        ranges.append((cursor, cursor + len(part), index))
        cursor += len(part) + 1
    matched: set[int] = set()
    for group in _SOURCE_EXECUTABLE_GROUPS:
        position = match.start(group)
        matched.update(index for start, end, index in ranges if start <= position < end)
    return tuple(sorted(matched))


def _instruction_line_indexes(
    lines: list[str],
    *,
    check: Callable[[], None] | None = None,
) -> tuple[
    set[int],
    tuple[_InstructionLexicalLine, ...],
    tuple[tuple[int, ...], ...],
]:
    """Find single-line and short split-line instruction payloads."""

    lexical_lines = _instruction_lexical_lines(lines, check=check)
    probes: list[str] = []
    for index, line in enumerate(lines):
        if check is not None and index % 128 == 0:
            check()
        probes.append(_instruction_probe(line, check=check))
    comment_membership = tuple(line.member for line in lexical_lines)
    comment_span_texts_list: list[tuple[str, ...]] = []
    for index, lexical in enumerate(lexical_lines):
        if check is not None and index % 128 == 0:
            check()
        span_texts: list[str] = []
        record_content = _instruction_record_content(lines[index])
        for span_index, span in enumerate(lexical.comment_spans):
            if check is not None and span_index % 128 == 0:
                check()
            span_texts.append(
                _instruction_probe(
                    record_content[span.start : span.end].strip(),
                    check=check,
                )
            )
        comment_span_texts_list.append(tuple(span_texts))
    comment_span_texts = tuple(comment_span_texts_list)
    comment_texts_list: list[str | None] = []
    contextual_probes_list: list[str] = []
    for index, (probe, lexical) in enumerate(zip(probes, lexical_lines, strict=True)):
        if check is not None and index % 128 == 0:
            check()
        comment_text = (
            " ".join(comment_span_texts[index])
            if lexical.comment_spans
            else probe.strip()
            if lexical.member
            else None
        )
        comment_texts_list.append(comment_text)
        contextual_probes_list.append(
            comment_text
            if comment_text is not None
            else _instruction_probe(lexical.unquoted_text, check=check)
        )
    comment_texts = tuple(comment_texts_list)
    contextual_probes = tuple(contextual_probes_list)
    executable_probes_list: list[str] = []
    for index, lexical in enumerate(lexical_lines):
        if check is not None and index % 128 == 0:
            check()
        executable_probes_list.append(
            _instruction_probe(
                _instruction_executable_text(lexical, check=check),
                check=check,
            )
        )
    executable_probes = tuple(executable_probes_list)
    executable_matches: set[int] = set()
    for width in (1, 2, 3):
        for start in range(0, len(executable_probes) - width + 1):
            if check is not None and start % 128 == 0:
                check()
            indexes = tuple(range(start, start + width))
            executable_matches.update(
                _instruction_executable_match_indexes(executable_probes, indexes)
            )
    single_contextual_matches: set[int] = set()
    single_standalone_matches: set[int] = set()
    for index, probe in enumerate(contextual_probes):
        if check is not None and index % 128 == 0:
            check()
        for pattern_index, pattern in enumerate(_SOURCE_CONTEXT_INSTRUCTION_PATTERNS):
            if check is not None and pattern_index % 8 == 0:
                check()
            if pattern.search(probe) is not None:
                single_contextual_matches.add(index)
                break
        if check is not None:
            check()
        if _instruction_has_standalone_instruction(
            ((comment_texts[index] or probe),),
            check=check,
        ):
            single_standalone_matches.add(index)
    matched = single_contextual_matches | single_standalone_matches | executable_matches
    # Prompt injections commonly split the authority claim and directive across
    # adjacent comments. Match bounded two/three-line windows while retaining the
    # original records for line-preserving replacement.
    for width in (2, 3):
        for start in range(0, len(probes) - width + 1):
            if check is not None and start % 128 == 0:
                check()
            indexes = tuple(range(start, start + width))
            single_indexes = tuple(index for index in indexes if index in single_contextual_matches)
            if single_indexes:
                complementary_indexes = tuple(
                    index for index in indexes if index not in single_contextual_matches
                )
                if not any(
                    comment_texts[index]
                    and _SOURCE_CONTEXT_PAYLOAD_FRAGMENT_PATTERN.search(comment_texts[index] or "")
                    is not None
                    for index in complementary_indexes
                ):
                    continue
            window = " ".join(contextual_probes[start : start + width])
            all_comment = all(comment_membership[index] for index in indexes)
            comment_label_authority = any(
                comment_texts[index] is not None
                and _instruction_has_label_authority(comment_texts[index] or "")
                for index in indexes
            )
            comment_authority = any(
                comment_texts[index] is not None
                and _instruction_has_authority(comment_texts[index] or "")
                for index in indexes
            )
            use_label_authority = (
                all_comment
                or comment_label_authority
                or any(_instruction_is_review_label(probes[index]) for index in indexes)
            )
            context_patterns = (
                _SOURCE_CONTEXT_INSTRUCTION_PATTERNS
                if use_label_authority
                else _SOURCE_STRONG_CONTEXT_INSTRUCTION_PATTERNS
            )
            has_context = False
            for pattern_index, pattern in enumerate(context_patterns):
                if check is not None and pattern_index % 8 == 0:
                    check()
                if pattern.search(window) is not None:
                    has_context = True
                    break
            nonblank_indexes = tuple(
                index for index in indexes if _instruction_content(probes[index]).strip()
            )
            standalone_comment_indexes = tuple(index for index in indexes if comment_texts[index])
            standalone_indexes = standalone_comment_indexes or nonblank_indexes
            if check is not None:
                check()
            has_standalone = bool(standalone_indexes) and _instruction_has_standalone_instruction(
                tuple(
                    (comment_texts[index] or contextual_probes[index])
                    for index in standalone_indexes
                ),
                check=check,
            )
            if not has_context:
                if not has_standalone:
                    continue
                standalone_single_indexes = tuple(
                    index for index in indexes if index in single_standalone_matches
                )
                if standalone_single_indexes:
                    standalone_complements = tuple(
                        index
                        for index in standalone_indexes
                        if index not in single_standalone_matches
                    )
                    if not any(
                        _instruction_has_standalone_fragment(
                            ((comment_texts[index] or contextual_probes[index]),),
                            check=check,
                        )
                        for index in standalone_complements
                    ):
                        continue
            if has_context and comment_authority:
                payload_indexes = tuple(
                    index
                    for index in indexes
                    if comment_texts[index]
                    and (
                        _instruction_has_authority(comment_texts[index] or "")
                        or _SOURCE_CONTEXT_PAYLOAD_FRAGMENT_PATTERN.search(
                            comment_texts[index] or ""
                        )
                        is not None
                        or (
                            has_standalone
                            and _instruction_has_standalone_fragment(
                                ((comment_texts[index] or contextual_probes[index]),),
                                check=check,
                            )
                        )
                    )
                )
            elif has_standalone and standalone_comment_indexes:
                payload_indexes = tuple(
                    index
                    for index in standalone_comment_indexes
                    if index in single_standalone_matches
                    or _instruction_has_standalone_fragment(
                        ((comment_texts[index] or contextual_probes[index]),),
                        check=check,
                    )
                )
            else:
                payload_indexes = tuple(
                    index for index in indexes if _instruction_content(probes[index]).strip()
                )
            if not payload_indexes:
                continue
            matched.update(payload_indexes)
    selected_spans: list[tuple[int, ...]] = []
    for index, line_span_texts in enumerate(comment_span_texts):
        if check is not None and index % 128 == 0:
            check()
        if index not in matched or not line_span_texts:
            selected_spans.append(())
            continue
        if index in executable_matches:
            selected_spans.append(())
            continue
        direct_list: list[int] = []
        for span_index, text in enumerate(line_span_texts):
            if check is not None and span_index % 128 == 0:
                check()
            contextual_match = False
            for pattern_index, pattern in enumerate(_SOURCE_CONTEXT_INSTRUCTION_PATTERNS):
                if check is not None and pattern_index % 8 == 0:
                    check()
                if pattern.search(text) is not None:
                    contextual_match = True
                    break
            if contextual_match or _instruction_has_standalone_instruction(
                (text,),
                check=check,
            ):
                direct_list.append(span_index)
        direct = tuple(direct_list)
        if direct:
            selected_spans.append(direct)
            continue
        fragments_list: list[int] = []
        for span_index, text in enumerate(line_span_texts):
            if check is not None and span_index % 128 == 0:
                check()
            if (
                _instruction_has_authority(text)
                or _SOURCE_CONTEXT_PAYLOAD_FRAGMENT_PATTERN.search(text) is not None
                or _instruction_has_standalone_fragment(
                    (text,),
                    check=check,
                )
            ):
                fragments_list.append(span_index)
        fragments = tuple(fragments_list)
        selected_spans.append(fragments or tuple(range(len(line_span_texts))))
    if check is not None:
        check()
    return matched, lexical_lines, tuple(selected_spans)


def neutralize_source_instructions(
    value: str,
    *,
    source: bool = False,
    check: Callable[[], None] | None = None,
    track_redacted_lines: bool = False,
    redacted_line_window: tuple[int, int] | None = None,
) -> SourceInstructionResult:
    """Neutralize reviewer-directed source text without changing line boundaries.

    ``source=True`` preserves executable text before a recognized comment marker.
    A matching executable line with no comment boundary is omitted in full because
    retaining only selected tokens would silently reinterpret untrusted source.
    """

    if check is not None:
        check()
    records = value.splitlines(keepends=True)
    line_values: list[str] = []
    for index, record in enumerate(records):
        if check is not None and index % 128 == 0:
            check()
        line_values.append(
            record[:-2]
            if record.endswith("\r\n")
            else record[:-1]
            if record.endswith(("\r", "\n"))
            else record
        )
    instruction_lines, lexical_lines, selected_spans = _instruction_line_indexes(
        line_values,
        check=check,
    )
    redacted = False
    incomplete = False
    redacted_lines: list[int] = []
    output: list[str] = []
    for line_number, (record, line) in enumerate(zip(records, line_values, strict=True), start=1):
        if check is not None and line_number % 128 == 0:
            check()
        if record.endswith("\r\n"):
            ending = "\r\n"
        elif record.endswith(("\r", "\n")):
            ending = record[-1:]
        else:
            ending = ""
        if line_number - 1 not in instruction_lines:
            output.append(record)
            continue
        redacted = True
        if track_redacted_lines and (
            redacted_line_window is None
            or redacted_line_window[0] <= line_number <= redacted_line_window[1]
        ):
            redacted_lines.append(line_number)
        has_diff_prefix = _instruction_has_diff_prefix(line)
        diff_prefix = line[:1] if has_diff_prefix else ""
        content = line[len(diff_prefix) :]
        if not source:
            output.append(f"{diff_prefix}{SOURCE_INSTRUCTION_REDACTION_MARKER}{ending}")
            continue
        lexical = lexical_lines[line_number - 1]
        span_indexes = selected_spans[line_number - 1]
        if not span_indexes:
            output.append(f"{diff_prefix}[untrusted source instruction line omitted]{ending}")
            incomplete = True
            continue
        pieces: list[str] = []
        preserved_parts: list[str] = []
        cursor = 0
        for span_index in span_indexes:
            span = lexical.comment_spans[span_index]
            preserved_part = content[cursor : span.start]
            pieces.append(preserved_part)
            preserved_parts.append(preserved_part)
            marker_separator = "" if not span.marker or span.marker.endswith(" ") else " "
            closer = f" {span.closer}" if span.closer is not None else ""
            pieces.append(
                f"{span.marker}{marker_separator}{SOURCE_INSTRUCTION_REDACTION_MARKER}{closer}"
            )
            cursor = span.end if span.end is not None else len(content)
        preserved_part = content[cursor:]
        pieces.append(preserved_part)
        preserved_parts.append(preserved_part)
        rewritten = "".join(pieces)
        output.append(f"{diff_prefix}{rewritten}{ending}")
        incomplete = incomplete or bool("".join(preserved_parts).strip())
    if check is not None:
        check()
    return SourceInstructionResult(
        text="".join(output),
        redacted=redacted,
        incomplete=incomplete,
        redacted_lines=tuple(redacted_lines),
    )


def private_key_spans(
    text: str, *, check: Callable[[], None] | None = None
) -> tuple[tuple[int, int, bool], ...]:
    """Return conservative PEM spans as ``(start, end, complete)`` tuples.

    A regex that stops at the first END marker is unsafe for nested or malformed
    input: it can hide an outer BEGIN from an unterminated-key fallback and leak
    the remaining key tail. This single linear marker scan tracks nesting and
    redacts through EOF whenever marker labels do not close in LIFO order.
    """

    if check is not None:
        check()
    spans: list[tuple[int, int, bool]] = []
    labels: list[str] = []
    span_start: int | None = None
    pending: tuple[bool, int, int] | None = None

    def first_line_break(start: int, end: int) -> int | None:
        carriage_return = text.find("\r", start, end)
        line_feed = text.find("\n", start, end)
        candidates = tuple(index for index in (carriage_return, line_feed) if index >= 0)
        return min(candidates) if candidates else None

    def consume_marker(
        is_begin: bool,
        marker_start: int,
        label: str,
        marker_end: int,
    ) -> bool:
        nonlocal span_start
        if _PRIVATE_KEY_LABEL.search(label) is None:
            return True
        if is_begin:
            if span_start is None:
                span_start = marker_start
            labels.append(label)
            return True
        if span_start is None:
            return True
        if not labels or labels[-1] != label:
            spans.append((span_start, len(text), False))
            return False
        labels.pop()
        if not labels:
            spans.append((span_start, marker_end, True))
            span_start = None
        return True

    for token_index, token in enumerate(_PEM_TOKEN.finditer(text), start=1):
        if check is not None and token_index % 128 == 0:
            check()
        token_start = token.start("token")
        token_end = token.end("token")
        if pending is not None:
            is_begin, marker_start, label_start = pending
            line_break = first_line_break(label_start, token_start)
            if line_break is None:
                if not consume_marker(
                    is_begin,
                    marker_start,
                    text[label_start:token_start],
                    token_start + _PEM_MARKER_CLOSE_LENGTH,
                ):
                    return tuple(spans)
            else:
                label = text[label_start:line_break]
                if _PRIVATE_KEY_LABEL.search(label) is not None:
                    if span_start is None and is_begin:
                        span_start = marker_start
                    if span_start is not None:
                        spans.append((span_start, len(text), False))
                        return tuple(spans)
            pending = None

        kind = token.group("kind")
        if kind is not None:
            pending = (kind.upper() == "BEGIN", token_start, token_end)

    if pending is not None:
        is_begin, marker_start, label_start = pending
        line_break = first_line_break(label_start, len(text))
        label_end = line_break if line_break is not None else len(text)
        label = text[label_start:label_end]
        if _PRIVATE_KEY_LABEL.search(label) is not None:
            if span_start is None and is_begin:
                span_start = marker_start
            if span_start is not None:
                spans.append((span_start, len(text), False))
                return tuple(spans)
    if span_start is not None:
        spans.append((span_start, len(text), False))
    if check is not None:
        check()
    return tuple(spans)


def secret_spans(text: str, *, check: Callable[[], None] | None = None) -> tuple[SecretSpan, ...]:
    """Return every secret span found against the original, unmodified text."""

    found: list[SecretSpan] = []
    for start, end, complete in private_key_spans(text, check=check):
        found.append(
            SecretSpan(
                kind="private-key-block" if complete else "private-key-prefix",
                start=start,
                end=end,
            )
        )
    for name, pattern in _SECRET_PATTERNS:
        if check is not None:
            check()
        for match_index, match in enumerate(pattern.finditer(text), start=1):
            found.append(SecretSpan(kind=name, start=match.start(), end=match.end()))
            if check is not None and match_index % 128 == 0:
                check()
    if check is not None:
        check()
    return tuple(found)


def redact_text(text: str) -> RedactionResult:
    found = secret_spans(text)
    if not found:
        return RedactionResult(text=text, blocked=False, matches=())

    counts: dict[str, int] = {}
    for secret in found:
        counts[secret.kind] = counts.get(secret.kind, 0) + 1
    matches = tuple(f"{name}:{count}" for name, count in counts.items())
    spans = sorted((secret.start, secret.end) for secret in found)

    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        pieces.append(text[cursor:start])
        pieces.append("[REDACTED_SECRET]")
        cursor = end
    pieces.append(text[cursor:])
    return RedactionResult(text="".join(pieces), blocked=True, matches=matches)
