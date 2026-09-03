"""Inspect ground-truth evidence for INV-0144.
Run:
    PYTHONPATH=src .venv/bin/python scripts/inspect_one_case.py
"""

from finance_controller.data.generator import generate
from finance_controller.matching.scorer import compute_evidence, confidence

dev, held = generate()
inv = next(i for i in held.invoices if i.invoice_id == "INV-0144")
true_match_id = next(
    lbl.right_id for lbl in held.match_labels if lbl.left_id == "INV-0144" and lbl.is_match
)
item = next(i for i in held.settlement_items if i.payment_id == true_match_id)

print(f"invoice:    customer_name={inv.customer_name!r}, amount={inv.amount}, "
      f"date={inv.invoice_date}, order_reference={inv.order_reference!r}")
print(f"settlement: customer_name={item.customer_name!r}, amount={item.amount}, "
      f"date={item.created_at}, order_reference={item.order_reference!r}")

ev = compute_evidence(inv, item)
print(f"\nevidence: vendor_sim={ev.vendor_similarity:.3f}, "
      f"amount_diff={ev.amount_difference}, date_diff={ev.date_difference_days}, "
      f"reference_status={ev.reference_status}")
print(f"confidence: {confidence(ev, inv.amount):.3f}")
