"""Generic PAN candidate generation utilities.

This module provides reusable building blocks for generating and filtering
candidate PANs from partially known patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

DigitLike = Union[int, str]
ConstraintKey = Union[int, str]
RawConstraints = (
    dict[int, set[int]]
    | dict[str, set[int]]
    | dict[ConstraintKey, set[int]]
)
GlobalConstraint = Callable[[str], bool]


@dataclass(frozen=True)
class PanPatternSpec:
    """Normalized PAN pattern specification.

    Attributes:
        raw_pattern: Original user-supplied pattern (may include separators).
        normalized_pattern: Digits and placeholder characters only.
        length: Number of PAN digits in the normalized pattern.
        fixed_digits: Index -> fixed digit mapping.
        variable_positions: Sorted positions that are placeholders.
    """

    raw_pattern: str
    normalized_pattern: str
    length: int
    fixed_digits: Dict[int, int]
    variable_positions: List[int]


def parse_pan_pattern(
    pattern: str,
    placeholders: Iterable[str] = ("X", "x", "?"),
    ignored_separators: Iterable[str] = (" ", "-"),
) -> PanPatternSpec:
    """Parse a PAN pattern into a normalized, reusable specification.

    Args:
        pattern: Pattern with fixed digits and placeholders, for example
            ``"4388 54XX XXXX 0665"`` or ``"438854XXXXXX0665"``.
        placeholders: Characters treated as unknown digits.
        ignored_separators: Characters removed before parsing.

    Returns:
        A :class:`PanPatternSpec` with fixed and variable positions.

    Raises:
        ValueError: If the pattern is empty after normalization or contains
            unsupported characters.
    """
    if not pattern:
        raise ValueError("pattern must be a non-empty string")

    separator_set = set(ignored_separators)
    placeholder_set = set(placeholders)

    normalized_chars: List[str] = [ch for ch in pattern if ch not in separator_set]
    if not normalized_chars:
        raise ValueError("pattern has no digits/placeholders after removing separators")

    fixed_digits: Dict[int, int] = {}
    variable_positions: List[int] = []

    for index, ch in enumerate(normalized_chars):
        if ch.isdigit():
            fixed_digits[index] = int(ch)
        elif ch in placeholder_set:
            variable_positions.append(index)
        else:
            raise ValueError(
                f"invalid character {ch!r} at normalized index {index}; "
                "use digits, placeholders, spaces, or hyphens"
            )

    normalized_pattern = "".join(normalized_chars)
    return PanPatternSpec(
        raw_pattern=pattern,
        normalized_pattern=normalized_pattern,
        length=len(normalized_pattern),
        fixed_digits=fixed_digits,
        variable_positions=variable_positions,
    )


def build_pattern_from_schema(
    length: int,
    fixed: Mapping[int, DigitLike],
    placeholder: str = "X",
) -> str:
    """Build a string pattern from a structured schema.

    Args:
        length: Total PAN length.
        fixed: Mapping of position index to fixed digit.
        placeholder: Placeholder character for unknown positions.

    Returns:
        A normalized pattern string (no spaces), suitable for
        :func:`generate_pan_candidates`.

    Raises:
        ValueError: If indices/digits are invalid.
    """
    if length <= 0:
        raise ValueError("length must be > 0")
    if not placeholder or len(placeholder) != 1:
        raise ValueError("placeholder must be a single character")

    chars = [placeholder for _ in range(length)]
    for index, digit in fixed.items():
        if index < 0 or index >= length:
            raise ValueError(f"fixed index {index} out of bounds for length {length}")
        parsed_digit = _coerce_digit(digit)
        chars[index] = str(parsed_digit)

    return "".join(chars)


def luhn_ok(pan: str) -> bool:
    """Return ``True`` when a PAN satisfies the Luhn checksum.

    This implementation works for arbitrary PAN lengths and expects a digits-only
    string.
    """
    if not pan or not pan.isdigit():
        return False

    total = 0
    for idx, ch in enumerate(reversed(pan)):
        digit = int(ch)
        if idx % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit

    return total % 10 == 0


def compute_luhn_check_digit(body: str) -> int:
    """Compute the Luhn check digit for a PAN body.

    Args:
        body: PAN without its final check digit (digits-only).

    Returns:
        The integer check digit in the range 0..9.

    Raises:
        ValueError: If ``body`` is empty or contains non-digit characters.
    """
    if not body:
        raise ValueError("body must be non-empty")
    if not body.isdigit():
        raise ValueError("body must contain only digits")

    total = 0
    for idx, ch in enumerate(reversed(body)):
        digit = int(ch)
        if idx % 2 == 0:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit

    return (10 - (total % 10)) % 10


def generate_pan_candidates(
    pattern: str,
    constraints: Optional[RawConstraints] = None,
    enforce_luhn: bool = True,
    max_candidates: Optional[int] = None,
    *,
    global_constraints: Optional[Sequence[GlobalConstraint]] = None,
    default_digits: Optional[set[int]] = None,
) -> List[str]:
    """Generate PAN candidates for a generic pattern and position constraints.

    Args:
        pattern: Pattern with fixed digits and placeholders, for example
            ``"438854XXXXXX0665"`` or ``"4XXX XXXX XXXX XXXX"``.
        constraints: Optional position-level allowed digit sets. Keys are
            zero-based indices in the normalized PAN. A special key
            ``"default"`` may be provided to define fallback digits for
            unconstrained placeholder positions.
        enforce_luhn: If ``True``, only return Luhn-valid candidates.
        max_candidates: Optional cap; generation stops once this many matches
            are collected.
        global_constraints: Optional global predicates applied to full PAN
            candidates (for example, issuer-range checks).
        default_digits: Allowed digits for unconstrained placeholder positions.
            Defaults to ``{0,1,2,3,4,5,6,7,8,9}``.

    Returns:
        List of candidate PANs matching the pattern and constraints.

    Raises:
        ValueError: If the pattern or constraints are invalid.
    """
    if max_candidates is not None and max_candidates <= 0:
        return []

    spec = parse_pan_pattern(pattern)
    pos_constraints, fallback_digits = _normalize_constraints(constraints, default_digits)

    _validate_allowed_digits(fallback_digits, label="default_digits")
    _validate_constraints(pos_constraints, spec.length)

    choices = _build_position_choices(spec, pos_constraints, fallback_digits)
    if choices is None:
        return []

    globals_list = list(global_constraints or [])

    direct = _try_direct_candidate(
        choices=choices,
        length=spec.length,
        enforce_luhn=enforce_luhn,
        global_constraints=globals_list,
    )
    if direct is not None:
        return direct[:max_candidates] if max_candidates is not None else direct

    if enforce_luhn:
        single_check_digit = _try_single_check_digit_fast_path(
            spec=spec,
            choices=choices,
            global_constraints=globals_list,
        )
        if single_check_digit is not None:
            return (
                single_check_digit[:max_candidates]
                if max_candidates is not None
                else single_check_digit
            )

    return _backtrack_generate(
        choices=choices,
        length=spec.length,
        enforce_luhn=enforce_luhn,
        global_constraints=globals_list,
        max_candidates=max_candidates,
    )


def _backtrack_generate(
    choices: Mapping[int, List[int]],
    length: int,
    enforce_luhn: bool,
    global_constraints: Sequence[GlobalConstraint],
    max_candidates: Optional[int],
) -> List[str]:
    """Generate candidates using depth-first enumeration over position choices."""
    results: List[str] = []
    mutable_digits: List[Optional[int]] = [None] * length

    def backtrack(position_cursor: int) -> None:
        if max_candidates is not None and len(results) >= max_candidates:
            return

        if position_cursor == length:
            candidate = "".join(str(digit) for digit in mutable_digits)
            if _candidate_matches(candidate, enforce_luhn, global_constraints):
                results.append(candidate)
            return

        for digit in choices[position_cursor]:
            mutable_digits[position_cursor] = digit
            backtrack(position_cursor + 1)
            if max_candidates is not None and len(results) >= max_candidates:
                return

    backtrack(0)
    return results


def _try_direct_candidate(
    choices: Mapping[int, List[int]],
    length: int,
    enforce_luhn: bool,
    global_constraints: Sequence[GlobalConstraint],
) -> Optional[List[str]]:
    """Return early when each position has a single deterministic digit."""
    if any(len(choices[index]) != 1 for index in range(length)):
        return None

    candidate = "".join(str(choices[index][0]) for index in range(length))
    return [candidate] if _candidate_matches(candidate, enforce_luhn, global_constraints) else []


def _try_single_check_digit_fast_path(
    spec: PanPatternSpec,
    choices: Mapping[int, List[int]],
    global_constraints: Sequence[GlobalConstraint],
) -> Optional[List[str]]:
    """Fast path when the only placeholder is the rightmost check digit."""
    last_index = spec.length - 1
    if spec.variable_positions != [last_index]:
        return None

    body = "".join(str(spec.fixed_digits[index]) for index in range(last_index))
    check_digit = compute_luhn_check_digit(body)
    if check_digit not in choices[last_index]:
        return []

    candidate = body + str(check_digit)
    return [candidate] if all(rule(candidate) for rule in global_constraints) else []


def _build_position_choices(
    spec: PanPatternSpec,
    pos_constraints: Mapping[int, set[int]],
    fallback_digits: set[int],
) -> Optional[Dict[int, List[int]]]:
    """Build per-position digit choices from fixed digits and constraints."""
    choices: Dict[int, List[int]] = {}
    for index in range(spec.length):
        if index in spec.fixed_digits:
            fixed_digit = spec.fixed_digits[index]
            if index in pos_constraints and fixed_digit not in pos_constraints[index]:
                return None
            choices[index] = [fixed_digit]
            continue

        allowed = pos_constraints.get(index, fallback_digits)
        if not (allowed_sorted := sorted(allowed)):
            return None
        choices[index] = allowed_sorted

    return choices


def _normalize_constraints(
    constraints: Optional[RawConstraints],
    default_digits: Optional[set[int]],
) -> tuple[Dict[int, set[int]], set[int]]:
    """Normalize raw constraints into position constraints and fallback digits."""
    fallback_digits = set(range(10)) if default_digits is None else set(default_digits)
    pos_constraints: Dict[int, set[int]] = {}

    if not constraints:
        return pos_constraints, fallback_digits

    for raw_key, allowed in constraints.items():
        if raw_key == "default":
            if default_digits is None:
                fallback_digits = set(allowed)
            continue

        if not isinstance(raw_key, int):
            raise ValueError(
                f"invalid constraint key {raw_key!r}; expected integer index or 'default'"
            )

        pos_constraints[raw_key] = set(allowed)

    return pos_constraints, fallback_digits


def _candidate_matches(
    candidate: str,
    enforce_luhn: bool,
    global_constraints: Sequence[GlobalConstraint],
) -> bool:
    """Apply Luhn and global predicate filters to a full candidate PAN."""
    if enforce_luhn and not luhn_ok(candidate):
        return False
    return all(rule(candidate) for rule in global_constraints)


def _validate_constraints(constraints: Mapping[int, set[int]], length: int) -> None:
    """Validate that index-level constraints are well-formed."""
    for index, allowed in constraints.items():
        if index < 0 or index >= length:
            raise ValueError(f"constraint index {index} out of bounds for length {length}")
        _validate_allowed_digits(allowed, label=f"constraints[{index}]")


def _validate_allowed_digits(values: set[int], label: str) -> None:
    """Validate a set of allowed digits."""
    if not values:
        return
    for digit in values:
        if not isinstance(digit, int) or digit < 0 or digit > 9:
            raise ValueError(f"{label} contains invalid digit {digit!r}; expected integers 0..9")


def _coerce_digit(digit: DigitLike) -> int:
    """Convert a digit-like value to an integer 0..9."""
    if isinstance(digit, int):
        parsed = digit
    elif isinstance(digit, str) and len(digit) == 1 and digit.isdigit():
        parsed = int(digit)
    else:
        raise ValueError(f"invalid digit value {digit!r}")

    if parsed < 0 or parsed > 9:
        raise ValueError(f"digit must be in range 0..9, got {parsed}")
    return parsed


__all__ = [
    "PanPatternSpec",
    "build_pattern_from_schema",
    "compute_luhn_check_digit",
    "generate_pan_candidates",
    "luhn_ok",
    "parse_pan_pattern",
]
