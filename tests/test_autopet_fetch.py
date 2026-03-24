from __future__ import annotations

from autopet_xai.fetch import select_autopet_fdg_cases, study_prefix_from_location


def test_study_prefix_from_location_strips_leading_dot() -> None:
    raw = "./FDG-PET-CT-Lesions/PETCT_123/01-01-2001-NA-Study/"
    assert study_prefix_from_location(raw) == "FDG-PET-CT-Lesions/PETCT_123/01-01-2001-NA-Study/"


def test_select_autopet_fdg_cases_balances_negative_and_positive() -> None:
    rows = [
        {"study_location": f"./FDG-PET-CT-Lesions/PETCT_NEG_{index:03d}/study/", "diagnosis": "NEGATIVE"}
        for index in range(40)
    ] + [
        {"study_location": f"./FDG-PET-CT-Lesions/PETCT_POS_{index:03d}/study/", "diagnosis": "MELANOMA"}
        for index in range(40)
    ]

    selected = select_autopet_fdg_cases(rows, target_count=16, seed=42)
    negative_count = sum(1 for row in selected if row["diagnosis"] == "NEGATIVE")
    positive_count = sum(1 for row in selected if row["diagnosis"] != "NEGATIVE")

    assert len(selected) == 16
    assert negative_count == 8
    assert positive_count == 8
