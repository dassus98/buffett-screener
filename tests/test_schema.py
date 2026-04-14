"""Unit tests for data_acquisition/schema.py.

Covers:
- LINE_ITEM_MAP structure (14 entries, valid enums, correct drop flags)
- LINE_ITEM_MAP internal consistency (buffett_name == dict key, non-empty notes)
- CANONICAL_COLUMNS identity mapping
- resolve_field() — ideal present, substitute fallback, priority ordering,
  None-value skip, DERIVED sentinel skip, negative values, MISSING/DROP|FLAG
- resolve_all_fields() — complete row, sparse row with missing critical fields
- get_drop_required_fields() — correct count, correct membership
- Data lineage — statement grouping matches DuckDB store DDL expectations

All tests are pure-Python with no I/O or network calls.
Hand-calculated expectations are documented inline for each test.
"""

from __future__ import annotations

import pytest

from data_acquisition.schema import (
    CANONICAL_COLUMNS,
    LINE_ITEM_MAP,
    LineItem,
    get_drop_required_fields,
    resolve_all_fields,
    resolve_field,
)


# ---------------------------------------------------------------------------
# LINE_ITEM_MAP structural tests
# ---------------------------------------------------------------------------


class TestLineItemMapStructure:
    """Verify LINE_ITEM_MAP contains the correct 14 entries with valid shapes."""

    EXPECTED_NAMES = {
        "net_income",
        "depreciation_amortization",
        "capital_expenditures",
        "total_revenue",
        "gross_profit",
        "sga",
        "operating_income",
        "interest_expense",
        "long_term_debt",
        "shareholders_equity",
        "eps_diluted",
        "shares_outstanding_diluted",
        "treasury_stock",
        "working_capital_change",
    }

    def test_exactly_14_entries(self) -> None:
        """LINE_ITEM_MAP must contain exactly 14 canonical fields."""
        assert len(LINE_ITEM_MAP) == 14

    def test_all_expected_names_present(self) -> None:
        """Every required buffett_name must be present as a key."""
        assert set(LINE_ITEM_MAP.keys()) == self.EXPECTED_NAMES

    def test_all_values_are_line_items(self) -> None:
        """Every value must be a LineItem instance."""
        for name, item in LINE_ITEM_MAP.items():
            assert isinstance(item, LineItem), f"{name} value is not a LineItem"

    def test_confidence_values_are_valid(self) -> None:
        """substitution_confidence must be one of High / Medium / Low."""
        valid = {"High", "Medium", "Low"}
        for name, item in LINE_ITEM_MAP.items():
            assert item.substitution_confidence in valid, (
                f"{name}.substitution_confidence={item.substitution_confidence!r} invalid"
            )

    def test_statement_values_are_valid(self) -> None:
        """statement must be one of the three recognised values."""
        valid = {"income_statement", "balance_sheet", "cash_flow"}
        for name, item in LINE_ITEM_MAP.items():
            assert item.statement in valid, (
                f"{name}.statement={item.statement!r} invalid"
            )

    def test_drop_if_missing_is_bool(self) -> None:
        """drop_if_missing must be a boolean."""
        for name, item in LINE_ITEM_MAP.items():
            assert isinstance(item.drop_if_missing, bool), (
                f"{name}.drop_if_missing is not bool"
            )

    def test_capital_expenditures_drop_required(self) -> None:
        """capital_expenditures must be drop_if_missing=True (needed for F1 and F12)."""
        assert LINE_ITEM_MAP["capital_expenditures"].drop_if_missing is True

    def test_sga_not_drop_required(self) -> None:
        """sga must be drop_if_missing=False (F8 is omitted, not fatal, when missing)."""
        assert LINE_ITEM_MAP["sga"].drop_if_missing is False

    def test_interest_expense_not_drop_required(self) -> None:
        """interest_expense must be drop_if_missing=False (zero-fill rule for debt-free cos)."""
        assert LINE_ITEM_MAP["interest_expense"].drop_if_missing is False

    def test_treasury_stock_not_drop_required(self) -> None:
        """treasury_stock must be drop_if_missing=False (buyback test omitted when absent)."""
        assert LINE_ITEM_MAP["treasury_stock"].drop_if_missing is False


# ---------------------------------------------------------------------------
# CANONICAL_COLUMNS tests
# ---------------------------------------------------------------------------


class TestCanonicalColumns:
    """Verify CANONICAL_COLUMNS maps every buffett_name to itself."""

    def test_same_keys_as_line_item_map(self) -> None:
        """CANONICAL_COLUMNS must have the same keys as LINE_ITEM_MAP."""
        assert set(CANONICAL_COLUMNS.keys()) == set(LINE_ITEM_MAP.keys())

    def test_identity_mapping(self) -> None:
        """Each key must map to the same string (buffett_name == column name)."""
        for name, column in CANONICAL_COLUMNS.items():
            assert column == name, (
                f"CANONICAL_COLUMNS[{name!r}]={column!r} — expected identity mapping"
            )


# ---------------------------------------------------------------------------
# resolve_field tests
# ---------------------------------------------------------------------------


class TestResolveFieldIdealPresent:
    """resolve_field returns ideal field value + 'High' confidence when ideal key exists."""

    def test_returns_ideal_value(self) -> None:
        """When ideal field is present, value must equal the raw_data entry."""
        raw = {"netIncome": 5_000_000}
        value, field_used, confidence = resolve_field(raw, "net_income")
        assert value == 5_000_000

    def test_returns_ideal_field_name(self) -> None:
        """field_used must be the ideal field name, not a substitute."""
        raw = {"netIncome": 5_000_000}
        _, field_used, _ = resolve_field(raw, "net_income")
        assert field_used == "netIncome"

    def test_returns_high_confidence_for_ideal(self) -> None:
        """Confidence must be 'High' when the ideal field is resolved."""
        raw = {"netIncome": 5_000_000}
        _, _, confidence = resolve_field(raw, "net_income")
        assert confidence == "High"

    def test_ideal_field_zero_value_resolves(self) -> None:
        """A value of 0 (not None) must resolve successfully via the ideal field."""
        # 0 is a valid financial figure (e.g. a breakeven year)
        raw = {"netIncome": 0}
        value, field_used, confidence = resolve_field(raw, "net_income")
        assert value == 0
        assert field_used == "netIncome"
        assert confidence == "High"

    def test_capital_expenditures_ideal(self) -> None:
        """CapEx ideal field resolves with High confidence."""
        raw = {"capitalExpenditure": -250_000}
        value, field_used, confidence = resolve_field(raw, "capital_expenditures")
        assert value == -250_000
        assert field_used == "capitalExpenditure"
        assert confidence == "High"


class TestResolveFieldSubstitutePresent:
    """resolve_field falls back to substitutes in priority order."""

    def test_first_substitute_used_when_ideal_absent(self) -> None:
        """When ideal is absent, the first substitute in the list is returned."""
        # net_income ideal = "netIncome"; first substitute = "netIncomeFromContinuingOperations"
        raw = {"netIncomeFromContinuingOperations": 4_800_000}
        value, field_used, confidence = resolve_field(raw, "net_income")
        assert value == 4_800_000
        assert field_used == "netIncomeFromContinuingOperations"

    def test_substitute_returns_item_confidence(self) -> None:
        """Confidence for a substitute must equal the item's substitution_confidence."""
        # net_income.substitution_confidence = "High"
        raw = {"netIncomeFromContinuingOperations": 4_800_000}
        _, _, confidence = resolve_field(raw, "net_income")
        assert confidence == LINE_ITEM_MAP["net_income"].substitution_confidence

    def test_second_substitute_used_when_first_absent(self) -> None:
        """If the first substitute is also absent, the second substitute is tried."""
        # net_income substitutes = ["netIncomeFromContinuingOperations", "Net Income", ...]
        raw = {"Net Income": 4_600_000}
        value, field_used, confidence = resolve_field(raw, "net_income")
        assert value == 4_600_000
        assert field_used == "Net Income"
        assert confidence == "High"

    def test_medium_confidence_substitute(self) -> None:
        """A Medium-confidence item's substitute returns 'Medium' confidence."""
        # shares_outstanding_diluted.substitution_confidence = "Medium"
        # substitutes[0] = "sharesOutstanding"
        raw = {"sharesOutstanding": 1_500_000_000}
        value, field_used, confidence = resolve_field(raw, "shares_outstanding_diluted")
        assert value == 1_500_000_000
        assert field_used == "sharesOutstanding"
        assert confidence == "Medium"

    def test_derived_sentinels_skipped(self) -> None:
        """DERIVED: sentinel strings must be skipped; resolve continues to next candidate."""
        # gross_profit substitutes = ["DERIVED:totalRevenue-costOfRevenue", "Gross Profit"]
        # The DERIVED sentinel should be skipped; "Gross Profit" (yfinance) should resolve.
        raw = {"Gross Profit": 8_000_000}
        value, field_used, confidence = resolve_field(raw, "gross_profit")
        assert value == 8_000_000
        assert field_used == "Gross Profit"
        assert confidence == "High"  # substitution_confidence of gross_profit item


class TestResolveFieldMissingDropRequired:
    """resolve_field returns (None, 'MISSING', 'DROP') when drop_if_missing=True."""

    def test_missing_drop_required_returns_none(self) -> None:
        """Value must be None when the field is absent and drop_if_missing=True."""
        raw: dict = {}
        value, _, _ = resolve_field(raw, "net_income")
        assert value is None

    def test_missing_drop_required_field_used_is_missing(self) -> None:
        """field_used must be 'MISSING'."""
        raw: dict = {}
        _, field_used, _ = resolve_field(raw, "net_income")
        assert field_used == "MISSING"

    def test_missing_drop_required_confidence_is_drop(self) -> None:
        """confidence must be 'DROP' when drop_if_missing=True."""
        raw: dict = {}
        _, _, confidence = resolve_field(raw, "net_income")
        assert confidence == "DROP"

    def test_capital_expenditures_missing_returns_drop(self) -> None:
        """CapEx is drop_if_missing=True — confirm DROP outcome."""
        raw: dict = {}
        value, field_used, confidence = resolve_field(raw, "capital_expenditures")
        assert value is None
        assert field_used == "MISSING"
        assert confidence == "DROP"

    def test_irrelevant_keys_in_raw_do_not_resolve(self) -> None:
        """Keys present in raw_data that don't match any candidate must not resolve."""
        raw = {"someOtherField": 99, "anotherField": 42}
        value, field_used, confidence = resolve_field(raw, "operating_income")
        assert value is None
        assert field_used == "MISSING"
        assert confidence == "DROP"


class TestResolveFieldMissingFlagOnly:
    """resolve_field returns (None, 'MISSING', 'FLAG') when drop_if_missing=False."""

    def test_missing_flag_only_returns_none(self) -> None:
        """Value must be None when field absent and drop_if_missing=False."""
        raw: dict = {}
        value, _, _ = resolve_field(raw, "sga")
        assert value is None

    def test_missing_flag_only_field_used_is_missing(self) -> None:
        """field_used must be 'MISSING'."""
        raw: dict = {}
        _, field_used, _ = resolve_field(raw, "sga")
        assert field_used == "MISSING"

    def test_missing_flag_only_confidence_is_flag(self) -> None:
        """confidence must be 'FLAG' (not 'DROP') when drop_if_missing=False."""
        raw: dict = {}
        _, _, confidence = resolve_field(raw, "sga")
        assert confidence == "FLAG"

    def test_interest_expense_missing_returns_flag(self) -> None:
        """interest_expense is drop_if_missing=False — confirm FLAG outcome."""
        raw: dict = {}
        value, field_used, confidence = resolve_field(raw, "interest_expense")
        assert value is None
        assert field_used == "MISSING"
        assert confidence == "FLAG"

    def test_treasury_stock_missing_returns_flag(self) -> None:
        """treasury_stock is drop_if_missing=False — confirm FLAG outcome."""
        raw: dict = {}
        _, _, confidence = resolve_field(raw, "treasury_stock")
        assert confidence == "FLAG"


class TestResolveFieldUnknownName:
    """resolve_field raises KeyError for unrecognised buffett_names."""

    def test_unknown_buffett_name_raises_key_error(self) -> None:
        """A buffett_name not in LINE_ITEM_MAP must raise KeyError."""
        with pytest.raises(KeyError):
            resolve_field({}, "this_field_does_not_exist")


# ---------------------------------------------------------------------------
# resolve_all_fields tests
# ---------------------------------------------------------------------------


class TestResolveAllFieldsComplete:
    """resolve_all_fields with a fully-populated mock row."""

    def _make_complete_row(self) -> dict:
        """Return a flat dict containing the ideal field for every LINE_ITEM_MAP entry."""
        return {
            "netIncome": 10_000_000,
            "depreciationAndAmortization": 1_500_000,
            "capitalExpenditure": -800_000,
            "totalRevenue": 50_000_000,
            "grossProfit": 20_000_000,
            "sellingGeneralAndAdministrativeExpenses": 5_000_000,
            "operatingIncome": 8_000_000,
            "interestExpense": 400_000,
            "longTermDebt": 3_000_000,
            "totalStockholdersEquity": 25_000_000,
            "epsdiluted": 2.50,
            "weightedAverageSharesDiluted": 4_000_000_000,
            "treasuryStock": 500_000,
            "changeInWorkingCapital": -200_000,
        }

    def test_returns_dict_with_all_14_keys(self) -> None:
        """Result must contain exactly one entry per LINE_ITEM_MAP key."""
        result = resolve_all_fields(self._make_complete_row())
        assert set(result.keys()) == set(LINE_ITEM_MAP.keys())

    def test_all_values_resolved(self) -> None:
        """Every value in result must be (non-None, field_name, 'High')."""
        result = resolve_all_fields(self._make_complete_row())
        for name, (value, field_used, confidence) in result.items():
            assert value is not None, f"{name}: expected non-None, got None"
            assert field_used != "MISSING", f"{name}: expected a field name, got MISSING"

    def test_net_income_resolved_correctly(self) -> None:
        """Spot-check: net_income resolves via ideal field with High confidence."""
        result = resolve_all_fields(self._make_complete_row())
        value, field_used, confidence = result["net_income"]
        assert value == 10_000_000
        assert field_used == "netIncome"
        assert confidence == "High"

    def test_eps_diluted_resolved_correctly(self) -> None:
        """Spot-check: eps_diluted resolves via ideal field."""
        result = resolve_all_fields(self._make_complete_row())
        value, field_used, _ = result["eps_diluted"]
        assert value == 2.50
        assert field_used == "epsdiluted"


class TestResolveAllFieldsMissingCritical:
    """resolve_all_fields with a row missing several required fields."""

    def _make_sparse_row(self) -> dict:
        """Return a row with only non-critical fields populated."""
        return {
            "sellingGeneralAndAdministrativeExpenses": 5_000_000,
            "treasuryStock": 500_000,
            "interestExpense": 400_000,
        }

    def test_drop_required_fields_return_drop(self) -> None:
        """Fields with drop_if_missing=True must return confidence='DROP' when absent."""
        result = resolve_all_fields(self._make_sparse_row())
        drop_fields = get_drop_required_fields()
        for name in drop_fields:
            _, _, confidence = result[name]
            assert confidence == "DROP", (
                f"{name}: expected 'DROP', got {confidence!r}"
            )

    def test_flag_only_fields_present_resolve_normally(self) -> None:
        """Fields with drop_if_missing=False that ARE present must resolve normally."""
        result = resolve_all_fields(self._make_sparse_row())
        # sga is present in our sparse row
        value, field_used, confidence = result["sga"]
        assert value == 5_000_000
        assert field_used == "sellingGeneralAndAdministrativeExpenses"

    def test_flag_only_fields_absent_return_flag(self) -> None:
        """Fields with drop_if_missing=False that are absent must return confidence='FLAG'."""
        # working_capital_change is absent from sparse row
        result = resolve_all_fields(self._make_sparse_row())
        _, _, confidence = result["working_capital_change"]
        assert confidence == "FLAG"

    def test_all_keys_still_present(self) -> None:
        """resolve_all_fields must always return all 14 keys, even when data is sparse."""
        result = resolve_all_fields(self._make_sparse_row())
        assert len(result) == 14


# ---------------------------------------------------------------------------
# get_drop_required_fields tests
# ---------------------------------------------------------------------------


class TestGetDropRequiredFields:
    """get_drop_required_fields returns the correct subset of buffett_names."""

    def test_returns_list(self) -> None:
        """Return type must be a list."""
        assert isinstance(get_drop_required_fields(), list)

    def test_net_income_in_drop_list(self) -> None:
        """net_income is drop_if_missing=True — must appear in drop list."""
        assert "net_income" in get_drop_required_fields()

    def test_sga_not_in_drop_list(self) -> None:
        """sga is drop_if_missing=False — must NOT appear in drop list."""
        assert "sga" not in get_drop_required_fields()

    def test_interest_expense_not_in_drop_list(self) -> None:
        """interest_expense is drop_if_missing=False — must NOT appear in drop list."""
        assert "interest_expense" not in get_drop_required_fields()

    def test_treasury_stock_not_in_drop_list(self) -> None:
        """treasury_stock is drop_if_missing=False — must NOT appear in drop list."""
        assert "treasury_stock" not in get_drop_required_fields()

    def test_working_capital_change_not_in_drop_list(self) -> None:
        """working_capital_change is drop_if_missing=False — must NOT appear."""
        assert "working_capital_change" not in get_drop_required_fields()

    def test_all_drop_fields_have_drop_if_missing_true(self) -> None:
        """Cross-check: every name returned by get_drop_required_fields() must have
        drop_if_missing=True in LINE_ITEM_MAP."""
        for name in get_drop_required_fields():
            assert LINE_ITEM_MAP[name].drop_if_missing is True, (
                f"{name} in drop list but drop_if_missing=False"
            )

    def test_all_non_drop_fields_absent_from_result(self) -> None:
        """Cross-check: fields with drop_if_missing=False must NOT appear in result."""
        drop_list = get_drop_required_fields()
        for name, item in LINE_ITEM_MAP.items():
            if not item.drop_if_missing:
                assert name not in drop_list, (
                    f"{name} has drop_if_missing=False but appears in drop list"
                )

    def test_exactly_10_drop_required_fields(self) -> None:
        """There must be exactly 10 fields with drop_if_missing=True.

        These 10 are the critical financial data points without which a
        security cannot be meaningfully analysed using Buffett's framework.
        """
        assert len(get_drop_required_fields()) == 10


# ---------------------------------------------------------------------------
# LINE_ITEM_MAP internal consistency
# ---------------------------------------------------------------------------


class TestLineItemMapConsistency:
    """Verify internal consistency between dict keys and LineItem attributes."""

    def test_buffett_name_matches_dict_key(self) -> None:
        """Each LineItem.buffett_name must be identical to its dict key.

        This prevents silent mismatches where a LineItem is filed under the
        wrong key, causing resolve_field to read the wrong spec.
        """
        for key, item in LINE_ITEM_MAP.items():
            assert item.buffett_name == key, (
                f"Key {key!r} has buffett_name={item.buffett_name!r}"
            )

    def test_all_items_have_non_empty_notes(self) -> None:
        """Every LineItem must document implementation caveats in notes.

        Notes are referenced by financials.py and data_quality.py for
        edge-case handling (sign conventions, derivation rules, zero-fill).
        """
        for name, item in LINE_ITEM_MAP.items():
            assert len(item.notes.strip()) > 0, (
                f"{name}.notes is empty — must describe implementation caveats"
            )

    def test_all_items_have_non_empty_ideal_field(self) -> None:
        """ideal_field must be a non-empty string (the primary API field name)."""
        for name, item in LINE_ITEM_MAP.items():
            assert isinstance(item.ideal_field, str) and len(item.ideal_field) > 0, (
                f"{name}.ideal_field is empty or not a string"
            )

    def test_substitutes_is_a_list(self) -> None:
        """substitutes must be a list (possibly empty, never None)."""
        for name, item in LINE_ITEM_MAP.items():
            assert isinstance(item.substitutes, list), (
                f"{name}.substitutes is {type(item.substitutes).__name__}, expected list"
            )


# ---------------------------------------------------------------------------
# Statement-type grouping (data lineage alignment with store.py DDL)
# ---------------------------------------------------------------------------


class TestStatementGrouping:
    """Verify that LINE_ITEM_MAP statement assignments align with store.py DDL.

    The DuckDB tables income_statement, balance_sheet, and cash_flow each
    contain a specific subset of canonical columns. These tests verify the
    grouping is consistent without importing store.py directly, so schema.py
    remains free of circular dependencies.
    """

    # Expected grouping — must match store.py _TABLE_COLUMNS
    EXPECTED_INCOME_STATEMENT = {
        "net_income", "total_revenue", "gross_profit", "sga",
        "operating_income", "interest_expense",
        "eps_diluted", "shares_outstanding_diluted",
    }
    EXPECTED_BALANCE_SHEET = {
        "long_term_debt", "shareholders_equity", "treasury_stock",
    }
    EXPECTED_CASH_FLOW = {
        "depreciation_amortization", "capital_expenditures",
        "working_capital_change",
    }

    def test_income_statement_fields(self) -> None:
        """8 fields must be assigned to 'income_statement'."""
        actual = {
            name for name, item in LINE_ITEM_MAP.items()
            if item.statement == "income_statement"
        }
        assert actual == self.EXPECTED_INCOME_STATEMENT

    def test_balance_sheet_fields(self) -> None:
        """3 fields must be assigned to 'balance_sheet'."""
        actual = {
            name for name, item in LINE_ITEM_MAP.items()
            if item.statement == "balance_sheet"
        }
        assert actual == self.EXPECTED_BALANCE_SHEET

    def test_cash_flow_fields(self) -> None:
        """3 fields must be assigned to 'cash_flow'."""
        actual = {
            name for name, item in LINE_ITEM_MAP.items()
            if item.statement == "cash_flow"
        }
        assert actual == self.EXPECTED_CASH_FLOW

    def test_statement_distribution_sums_to_14(self) -> None:
        """8 + 3 + 3 = 14 — every field assigned to exactly one statement."""
        counts = {}
        for item in LINE_ITEM_MAP.values():
            counts[item.statement] = counts.get(item.statement, 0) + 1
        assert sum(counts.values()) == 14
        assert counts["income_statement"] == 8
        assert counts["balance_sheet"] == 3
        assert counts["cash_flow"] == 3


# ---------------------------------------------------------------------------
# resolve_field edge-case tests
# ---------------------------------------------------------------------------


class TestResolveFieldEdgeCases:
    """Additional edge-case tests for resolve_field()."""

    def test_none_value_in_raw_data_is_skipped(self) -> None:
        """A field present in raw_data with value=None must be skipped.

        resolve_field checks ``raw_data[candidate] is not None``, so an
        explicit None should fall through to the next candidate or MISSING.
        """
        # Ideal field exists but is None — should skip to substitute
        raw = {"netIncome": None, "netIncomeFromContinuingOperations": 3_000_000}
        value, field_used, confidence = resolve_field(raw, "net_income")
        assert value == 3_000_000
        assert field_used == "netIncomeFromContinuingOperations"

    def test_all_candidates_none_returns_missing(self) -> None:
        """If every candidate key is present but None, result is MISSING."""
        raw = {"netIncome": None, "netIncomeFromContinuingOperations": None,
               "Net Income": None, "Net Income From Continuing Operations": None}
        value, field_used, confidence = resolve_field(raw, "net_income")
        assert value is None
        assert field_used == "MISSING"
        assert confidence == "DROP"

    def test_ideal_preferred_over_substitute_when_both_present(self) -> None:
        """When both ideal and substitute exist, ideal must win.

        Priority order: ideal first, then substitutes in list order.
        """
        raw = {
            "netIncome": 10_000_000,  # ideal — should be selected
            "netIncomeFromContinuingOperations": 9_500_000,  # substitute
        }
        value, field_used, confidence = resolve_field(raw, "net_income")
        assert value == 10_000_000
        assert field_used == "netIncome"
        assert confidence == "High"

    def test_negative_net_income_resolves_correctly(self) -> None:
        """A loss year (negative net income) must resolve with the value as-is.

        Negative financial values are valid — they represent loss years, cash
        outflows, or share reductions. resolve_field must not filter them.
        """
        raw = {"netIncome": -2_500_000}
        value, field_used, confidence = resolve_field(raw, "net_income")
        assert value == -2_500_000
        assert field_used == "netIncome"
        assert confidence == "High"

    def test_zero_long_term_debt_resolves(self) -> None:
        """A debt-free company with long_term_debt=0 must resolve normally.

        Zero is a valid non-None value and must not be skipped. This is
        important for F5 (debt payoff = 0 years → automatic pass) and
        the interest_expense zero-fill rule.
        """
        raw = {"longTermDebt": 0}
        value, field_used, confidence = resolve_field(raw, "long_term_debt")
        assert value == 0
        assert field_used == "longTermDebt"
        assert confidence == "High"

    def test_float_value_resolves(self) -> None:
        """Float values (common for EPS) resolve correctly."""
        raw = {"epsdiluted": 3.42}
        value, field_used, confidence = resolve_field(raw, "eps_diluted")
        assert value == 3.42
        assert field_used == "epsdiluted"

    def test_substitute_priority_order_respected(self) -> None:
        """When multiple substitutes exist, the first in the list wins.

        This tests the priority chain: if both the 2nd and 3rd substitutes
        are present, the 2nd (higher priority) must be selected.
        """
        # For net_income: subs = [netIncomeFromContinuingOperations, Net Income, ...]
        # Provide only 2nd and 3rd substitutes; 2nd should win.
        raw = {
            "Net Income": 4_000_000,
            "Net Income From Continuing Operations": 3_800_000,
        }
        value, field_used, _ = resolve_field(raw, "net_income")
        # "Net Income" is subs[1], "Net Income From Continuing Operations" is subs[2]
        # subs[1] comes first in iteration → should be selected
        assert field_used == "Net Income"
        assert value == 4_000_000
