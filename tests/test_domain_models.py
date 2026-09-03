"""Smoke tests for the domain model.

These exist to catch structural mistakes early — before the generator
or matching engine are built on top of this — not to test business logic
(there isn't any yet).
"""

from datetime import date
from decimal import Decimal

from finance_controller.domain.enums import EntityType, MatchStatus, Source
from finance_controller.domain.models import (
    BankCredit,
    Invoice,
    MatchEvidence,
    MatchLabel,
    MatchPrediction,
    SettlementItem,
)


def test_invoice_constructs() -> None:
    invoice = Invoice(
        invoice_id="INV-1001",
        order_reference="ORD-5001",
        customer_name="ABC Technologies Pvt Ltd",
        amount=Decimal("50000.00"),
        invoice_date=date(2026, 8, 1),
    )
    assert invoice.amount == Decimal("50000.00")


def test_settlement_item_links_to_invoice_via_order_reference() -> None:
    invoice = Invoice(
        invoice_id="INV-1001",
        order_reference="ORD-5001",
        customer_name="ABC Technologies Pvt Ltd",
        amount=Decimal("50000.00"),
        invoice_date=date(2026, 8, 1),
    )
    item = SettlementItem(
        entity_id="pay_ABC123",
        entity_type=EntityType.PAYMENT,
        order_reference="ORD-5001",
        payment_id="pay_ABC123",
        customer_name="ABC TECH PVT LTD",
        amount=Decimal("50000.00"),
        fee=Decimal("1000.00"),
        tax=Decimal("180.00"),
        net_amount=Decimal("48820.00"),
        settlement_id="setl_XYZ789",
        settlement_utr="UTR000111222",
        created_at=date(2026, 8, 1),
    )
    assert item.order_reference == invoice.order_reference


def test_settlement_group_sums_to_bank_credit() -> None:
    """Two settlement items sharing a settlement_id should net to the
    bank credit amount — this is the arithmetic check that replaces a
    subset-sum search in stage 2 of the reconciliation loop."""
    item_a = SettlementItem(
        entity_id="pay_A",
        entity_type=EntityType.PAYMENT,
        order_reference="ORD-1",
        payment_id="pay_A",
        customer_name="Vendor A",
        amount=Decimal("10000.00"),
        fee=Decimal("200.00"),
        tax=Decimal("36.00"),
        net_amount=Decimal("9764.00"),
        settlement_id="setl_1",
        settlement_utr="UTR001",
        created_at=date(2026, 8, 1),
    )
    item_b = SettlementItem(
        entity_id="pay_B",
        entity_type=EntityType.PAYMENT,
        order_reference="ORD-2",
        payment_id="pay_B",
        customer_name="Vendor B",
        amount=Decimal("5000.00"),
        fee=Decimal("100.00"),
        tax=Decimal("18.00"),
        net_amount=Decimal("4882.00"),
        settlement_id="setl_1",
        settlement_utr="UTR001",
        created_at=date(2026, 8, 1),
    )
    credit = BankCredit(
        credit_id="BC-1",
        utr="UTR001",
        amount=Decimal("14646.00"),
        value_date=date(2026, 8, 3),
    )
    assert item_a.net_amount + item_b.net_amount == credit.amount


def test_match_prediction_carries_typed_evidence_not_free_text() -> None:
    evidence = MatchEvidence(
        vendor_similarity=0.94,
        amount_difference=Decimal("0.00"),
        date_difference_days=1,
        reference_status="match",
    )
    prediction = MatchPrediction(
        left_id="INV-1001",
        right_id="pay_ABC123",
        confidence=0.97,
        status=MatchStatus.AUTO_MATCHED,
        evidence=evidence,
        reason="Vendor similarity 94%, exact amount, reference match, 1 day apart.",
    )
    assert prediction.status is MatchStatus.AUTO_MATCHED
    assert prediction.evidence.vendor_similarity == 0.94


def test_match_label_is_source_aware() -> None:
    label = MatchLabel(
        left_id="INV-1001",
        left_source=Source.INVOICE,
        right_id="pay_ABC123",
        right_source=Source.SETTLEMENT_ITEM,
        is_match=True,
    )
    assert label.is_match is True
