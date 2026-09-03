"""Unit tests for matching pipeline."""

from decimal import Decimal

from finance_controller.data.generator import generate
from finance_controller.domain.enums import ExceptionReason, MatchStatus
from finance_controller.domain.models import Invoice, SettlementItem
from finance_controller.evaluation.metrics import evaluate
from finance_controller.matching.calibration import calibrate
from finance_controller.matching.candidates import generate_candidates
from finance_controller.matching.engine import reconcile_all, reconcile_invoice
from finance_controller.matching.scorer import (
    _normalize_reference,
    _normalize_vendor,
    vendor_similarity,
)
from finance_controller.matching.stage2 import reconcile_settlements


def test_normalize_vendor_and_reference() -> None:
    assert _normalize_vendor("ABC Technologies Pvt Ltd") == "ABC TECHNOLOGIES"
    assert _normalize_reference("ORD-0007") == "0007"
    assert _normalize_reference("PO-0007") == "0007"
    assert _normalize_reference(None) is None


def test_vendor_similarity_and_scoring() -> None:
    sim = vendor_similarity("ABC Technologies Pvt Ltd", "ABC TECH PVT LTD")
    assert sim > 0.60


def test_no_leakage_in_matching_modules() -> None:
    """Ensure matching modules do not import or access MatchLabel."""
    import finance_controller.matching.candidates as cand
    import finance_controller.matching.engine as eng
    import finance_controller.matching.scorer as scr
    import finance_controller.matching.stage2 as st2

    for mod in (cand, scr, eng, st2):
        assert "MatchLabel" not in dir(mod)


def test_candidate_generation_finds_dropped_reference() -> None:
    from datetime import date

    inv = Invoice(
        invoice_id="INV-100",
        order_reference="ORD-100",
        customer_name="Kestrel Media Private Limited",
        amount=Decimal("15000.00"),
        invoice_date=date(2026, 8, 1),
    )
    item = SettlementItem(
        entity_id="pay_100",
        entity_type="payment",  # type: ignore[arg-type]
        order_reference=None,  # dropped reference
        payment_id="pay_100",
        customer_name="KESTREL MEDIA PVT LTD",
        amount=Decimal("15000.00"),
        fee=Decimal("300.00"),
        tax=Decimal("54.00"),
        net_amount=Decimal("14646.00"),
        settlement_id="SETL-100",
        settlement_utr="UTR100",
        created_at=date(2026, 8, 2),
    )

    cands = generate_candidates([inv], [item])
    assert len(cands["INV-100"]) == 1
    assert cands["INV-100"][0].settlement_item.payment_id == "pay_100"


def test_ambiguity_detection_in_reconcile_invoice() -> None:
    from datetime import date

    inv = Invoice(
        invoice_id="INV-AMBIG",
        order_reference="ORD-800",
        customer_name="Sundar Spices Pvt Ltd",
        amount=Decimal("25000.00"),
        invoice_date=date(2026, 8, 1),
    )
    item_a = SettlementItem(
        entity_id="pay_A",
        entity_type="payment",  # type: ignore[arg-type]
        order_reference="ORD-800",
        payment_id="pay_A",
        customer_name="SUNDAR SPICES PVT LTD",
        amount=Decimal("25000.00"),
        fee=Decimal("500.00"),
        tax=Decimal("90.00"),
        net_amount=Decimal("24410.00"),
        settlement_id="SETL-80",
        settlement_utr="UTR80",
        created_at=date(2026, 8, 2),
    )
    item_b = SettlementItem(
        entity_id="pay_B",
        entity_type="payment",  # type: ignore[arg-type]
        order_reference="PO-800",
        payment_id="pay_B",
        customer_name="SUNDAR SPICES PVT LTD",
        amount=Decimal("25000.00"),
        fee=Decimal("500.00"),
        tax=Decimal("90.00"),
        net_amount=Decimal("24410.00"),
        settlement_id="SETL-80",
        settlement_utr="UTR80",
        created_at=date(2026, 8, 2),
    )

    dev, _ = generate()
    thresholds = calibrate(dev.invoices, dev.settlement_items, dev.match_labels)

    pred = reconcile_invoice(inv, [item_a, item_b], thresholds)
    assert pred.status == MatchStatus.REVIEW
    assert pred.exception_reason == ExceptionReason.AMBIGUOUS_CANDIDATES


def test_end_to_end_reconciliation_check() -> None:
    dev, held = generate()
    thresholds = calibrate(dev.invoices, dev.settlement_items, dev.match_labels)
    assert thresholds.auto_match > thresholds.review_floor

    predictions = reconcile_all(held.invoices, held.settlement_items, thresholds)
    metrics = evaluate(predictions, held.match_labels)

    assert metrics.total == len(held.invoices)
    assert metrics.auto_match_precision > 0.85
    assert metrics.recall > 0.60

    stage2_results = reconcile_settlements(held.settlement_items, held.bank_credits)
    assert len(stage2_results) > 0


def test_decoy_score_ceiling_is_below_any_plausible_auto_match_threshold() -> None:
    """Provable, not empirical: given current weights and the generator's
    minimum settlement delay of 1 day, a decoy's theoretical maximum score
    is 0.6871. If this test ever fails, either the weights changed or the
    generator's date-drift minimum changed — both are load-bearing for the
    'decoys structurally cannot auto-match' guarantee."""
    from finance_controller.matching.scorer import (
        DATE_SCORE_HALF_LIFE_DAYS,
        W_AMOUNT,
        W_DATE,
        W_REFERENCE,
        W_VENDOR,
    )

    theoretical_max_date_score = float(0.5 ** (1 / DATE_SCORE_HALF_LIFE_DAYS))
    theoretical_max_decoy_score = (
        W_AMOUNT * 1.0
        + W_VENDOR * 1.0
        + W_REFERENCE * (-1.0)
        + W_DATE * theoretical_max_date_score
    )
    assert theoretical_max_decoy_score < 0.70
