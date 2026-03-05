"""Tests for generic PAN candidate generation utilities."""

import pytest

from ocr_service.modules.pan_candidates import (
    build_pattern_from_schema,
    compute_luhn_check_digit,
    generate_pan_candidates,
    luhn_ok,
    parse_pan_pattern,
)


def test_luhn_ok_known_valid_and_invalid() -> None:
    assert luhn_ok("4111111111111111") is True
    assert luhn_ok("4111111111111112") is False


def test_compute_luhn_check_digit_known_value() -> None:
    # Canonical Luhn sample: 7992739871 + 3 -> 79927398713
    assert compute_luhn_check_digit("7992739871") == 3


def test_compute_luhn_check_digit_rejects_non_digits() -> None:
    with pytest.raises(ValueError):
        compute_luhn_check_digit("79927A")


def test_parse_pan_pattern_supports_spaces_and_x() -> None:
    spec = parse_pan_pattern("4XXX XXXX XXXX XXXX")
    assert spec.length == 16
    assert spec.fixed_digits == {0: 4}
    assert len(spec.variable_positions) == 15


def test_build_pattern_from_schema_generates_expected_string() -> None:
    pattern = build_pattern_from_schema(
        length=16,
        fixed={0: 4, 1: 3, 2: 8, 3: 8, 12: 0, 13: 6, 14: 6, 15: 5},
    )
    assert pattern == "4388XXXXXXXX0665"


def test_generate_pan_candidates_no_luhn_simple_case() -> None:
    candidates = generate_pan_candidates("12X", enforce_luhn=False)
    assert len(candidates) == 10
    assert candidates[0] == "120"
    assert candidates[-1] == "129"


def test_generate_pan_candidates_applies_position_constraints() -> None:
    # Only position 1 varies and is constrained.
    candidates = generate_pan_candidates(
        "4X11",
        constraints={1: {3, 8}},
        enforce_luhn=False,
    )
    assert candidates == ["4311", "4811"]


def test_generate_pan_candidates_luhn_single_unknown_check_digit() -> None:
    # Exactly one Luhn-valid completion should exist for this body.
    candidates = generate_pan_candidates("7992739871X", enforce_luhn=True)
    assert candidates == ["79927398713"]


def test_generate_pan_candidates_single_check_digit_respects_constraints() -> None:
    candidates = generate_pan_candidates(
        "7992739871X",
        constraints={10: {3, 5}},
        enforce_luhn=True,
    )
    assert candidates == ["79927398713"]


def test_generate_pan_candidates_single_check_digit_disallowed_returns_empty() -> None:
    candidates = generate_pan_candidates(
        "7992739871X",
        constraints={10: {5}},
        enforce_luhn=True,
    )
    assert candidates == []


def test_generate_pan_candidates_no_placeholders_direct_candidate() -> None:
    assert generate_pan_candidates("4111111111111111", enforce_luhn=True) == [
        "4111111111111111"
    ]
    assert generate_pan_candidates("4111111111111112", enforce_luhn=True) == []


def test_generate_pan_candidates_single_check_digit_without_luhn_keeps_all() -> None:
    candidates = generate_pan_candidates("7992739871X", enforce_luhn=False)
    assert len(candidates) == 10


def test_generate_pan_candidates_supports_global_constraints() -> None:
    candidates = generate_pan_candidates(
        "4XX",
        enforce_luhn=False,
        global_constraints=[lambda pan: int(pan[-1]) % 2 == 0],
    )
    assert all(int(c[-1]) % 2 == 0 for c in candidates)
    assert len(candidates) == 50


def test_generate_pan_candidates_supports_default_key_in_constraints() -> None:
    candidates = generate_pan_candidates(
        "12XX",
        constraints={"default": {7}, 2: {3, 4}},
        enforce_luhn=False,
    )
    assert candidates == ["1237", "1247"]


def test_default_digits_argument_overrides_constraints_default_key() -> None:
    candidates = generate_pan_candidates(
        "12XX",
        constraints={"default": {7}, 2: {3, 4}},
        default_digits={9},
        enforce_luhn=False,
    )
    assert candidates == ["1239", "1249"]


def test_example_pattern_4388_54xx_xxxx_0665_with_constraints() -> None:
    pattern = "438854XXXXXX0665"
    constraints = {
        6: {0, 6},
        11: {0, 5, 6},
    }

    candidates = generate_pan_candidates(
        pattern,
        constraints=constraints,
        enforce_luhn=True,
        max_candidates=50,
    )

    assert len(candidates) == 50
    assert all(len(candidate) == 16 for candidate in candidates)
    assert all(candidate.startswith("438854") and candidate.endswith("0665") for candidate in candidates)
    assert all(candidate[6] in {"0", "6"} for candidate in candidates)
    assert all(candidate[11] in {"0", "5", "6"} for candidate in candidates)
    assert all(luhn_ok(candidate) for candidate in candidates)
