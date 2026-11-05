"""
Generate realistic commercial B2B/D2C supply chain & defect dataset.
Produces:
- data/commercial_case_study/transactions.csv
- data/commercial_case_study/returns_notes.csv
"""

import random
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

random.seed(42)

OUT_DIR = Path("data/commercial_case_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["Electronics", "Textiles_Apparel", "Industrial_Hardware", "Cosmetics", "Furniture"]
REGIONS = ["Metro_North", "Metro_South", "West_Coast", "East_Coast", "Central_Hub"]
STATUSES = ["Delivered", "Delivered", "Delivered", "Returned", "Canceled"]

COMPLAINT_TEMPLATES = [
    # Cluster 1: Damage
    (1, "The cardboard carton arrived crushed and the interior display was shattered. Completely unusable."),
    (1, "Box was severely dented during transit, product chassis is bent and broken. Demanding a refund."),
    (1, "Liquid leaked all over the packaging, ruined the other goods in the box."),
    (2, "Corner of the furniture arrived chipped and scratched due to insufficient bubble wrap."),
    (1, "Item was delivered broken in half. Package was dropped by courier."),

    # Cluster 2: Missing Parts
    (2, "The power brick and charging cable were missing from the box."),
    (2, "Hardware package arrived incomplete. Screws, bolts and mounting bracket were absent."),
    (2, "Only received 2 units out of the 4 invoiced on the delivery challan."),
    (3, "Remote control was missing from the package, cannot operate unit."),

    # Cluster 3: Delay
    (2, "Shipment arrived 14 days past the committed deadline. Our factory assembly line was stalled."),
    (1, "Took over three weeks past estimated delivery date. Had to procure from local vendor at higher cost."),
    (2, "Delayed delivery caused us to miss our seasonal promotional window."),
    (3, "Courier tracking showed no movement for 10 days without any notification."),

    # Cluster 4: Wrong Item
    (2, "Ordered 220V European standard unit, but received 110V US model. Cannot use in our facility."),
    (3, "Shipped wrong color and wrong dimension. Ordered Black XL, received Grey Medium."),
    (2, "Catalog model number does not match what was delivered. Seller sent obsolete variant."),

    # Cluster 5: Defective / Quality
    (1, "Defective out of the box. Power supply trips the circuit breaker immediately upon plugging in."),
    (1, "Material quality is sub-par. Textile seams came apart on the first day of inspection."),
    (2, "Hydraulic pump overheats within 10 minutes of operation. Suspect counterfeit or factory second."),
    (1, "Motor seized after two hours of light duty. Requesting immediate replacement under warranty.")
]

def generate_data(num_tx=500, num_notes=160):
    start_date = datetime(2025, 1, 1)
    
    tx_records = []
    for i in range(1, num_tx + 1):
        tx_id = f"INV-2025-{i:05d}"
        client_id = f"CLT-{random.randint(100, 350)}"
        cat = random.choice(CATEGORIES)
        reg = random.choice(REGIONS)
        
        if cat == "Electronics":
            amt = round(random.uniform(500.0, 3200.0), 2)
            freight = round(random.uniform(25.0, 80.0), 2)
        elif cat == "Industrial_Hardware":
            amt = round(random.uniform(400.0, 2500.0), 2)
            freight = round(random.uniform(60.0, 220.0), 2)
        elif cat == "Furniture":
            amt = round(random.uniform(300.0, 1800.0), 2)
            freight = round(random.uniform(80.0, 250.0), 2)
        elif cat == "Textiles_Apparel":
            amt = round(random.uniform(150.0, 750.0), 2)
            freight = round(random.uniform(15.0, 45.0), 2)
        else:
            amt = round(random.uniform(60.0, 300.0), 2)
            freight = round(random.uniform(10.0, 30.0), 2)
            
        status = random.choice(STATUSES)
        delay = random.choice([0, 0, 0, 0, 1, 2, 4, 8, 14, 21])
        tx_date = start_date + timedelta(days=random.randint(0, 180))
        
        tx_records.append({
            "invoice_no": tx_id,
            "client_id": client_id,
            "product_category": cat,
            "region": reg,
            "invoice_amount": amt,
            "shipping_fee": freight,
            "delivery_status": status,
            "invoice_date": tx_date.strftime("%Y-%m-%d %H:%M:%S"),
            "days_late": delay
        })
        
    df_tx = pd.DataFrame(tx_records)
    df_tx.to_csv(OUT_DIR / "transactions.csv", index=False)
    print(f"Generated {len(df_tx)} transactions -> {OUT_DIR / 'transactions.csv'}")

    # Generate Feedback Notes
    note_records = []
    # Pick a subset of transactions to have complaints (heavily weighted towards returned/late orders)
    candidate_tx = df_tx[(df_tx["delivery_status"].isin(["Returned", "Canceled"])) | (df_tx["days_late"] > 3)]
    candidate_ids = candidate_tx["invoice_no"].tolist()
    
    # Also add some random normal delivered orders
    delivered_ids = df_tx[df_tx["delivery_status"] == "Delivered"]["invoice_no"].tolist()
    sample_ids = candidate_ids + random.sample(delivered_ids, min(50, len(delivered_ids)))
    random.shuffle(sample_ids)
    
    for idx, tx_id in enumerate(sample_ids[:num_notes], 1):
        note_id = f"NOTE-{idx:04d}"
        score, text_template = random.choice(COMPLAINT_TEMPLATES)
        # Add slight variations
        prefix = random.choice(["URGENT: ", "Complaint: ", "Quality Return: ", "Defect Report: ", ""])
        note_text = prefix + text_template
        note_date = start_date + timedelta(days=random.randint(5, 190))
        
        note_records.append({
            "complaint_id": note_id,
            "invoice_no": tx_id,
            "severity_rating": score,
            "customer_remarks": note_text,
            "complaint_date": note_date.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    df_notes = pd.DataFrame(note_records)
    df_notes.to_csv(OUT_DIR / "returns_notes.csv", index=False)
    print(f"Generated {len(df_notes)} return notes -> {OUT_DIR / 'returns_notes.csv'}")

if __name__ == "__main__":
    generate_data()

