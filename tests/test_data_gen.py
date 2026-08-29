import csv
import random
from datetime import datetime, timedelta

WEEKS = 60
TOTAL_DAYS = WEEKS * 7
REGIONS_5 = ["North", "South", "East", "West", "Central"]
CATEGORIES_5 = ["Electronics", "Apparel", "Home", "Sports", "Books"]
START_DATE = datetime(2024, 1, 1)

FEEDBACK_TEMPLATES = [
    "The package arrived on time and in great condition.",
    "Shipping took way longer than expected this week.",
    "Item quality exceeded my expectations completely.",
    "Box was damaged upon arrival but product is okay.",
    "Customer service resolved my issue very quickly.",
    "Will definitely order again from this store.",
    "Price is a bit high for what you get.",
    "Missing items from the order when it arrived.",
    "Tracking number never updated after dispatch.",
    "Very satisfied with the overall experience."
]

def base_row(day_idx, order_id, regions=REGIONS_5, cats=CATEGORIES_5):
    d = START_DATE + timedelta(days=day_idx)
    return {
        "order_id": order_id,
        "order_date": d.strftime("%Y-%m-%d"),
        "region": random.choice(regions),
        "category": random.choice(cats),
        "amount": round(random.gammavariate(3.0, 15.0), 2),
        "quantity": random.choice([1, 1, 2, 2, 3]),
        "customer_feedback": random.choice(FEEDBACK_TEMPLATES)
    }

def generate_scenario(name):
    rows = []
    oid = 100000
    
    for day_idx in range(TOTAL_DAYS):
        daily_count = random.randint(43, 85)
        for _ in range(daily_count):
            r = base_row(day_idx, f"ORD-{oid}")
            oid += 1
            
            week = day_idx // 7
            
            # 1. pure_noise
            if name == "pure_noise.csv":
                pass
                
            # 2. localised_event (Weeks 20-28, North degrades)
            elif name == "localised_event.csv":
                if 20 <= week <= 28 and r["region"] == "North":
                    r["amount"] = round(r["amount"] * 0.55, 2)
                    r["customer_feedback"] = "Package delayed and damaged upon arrival."
                    
            # 3. pure_mix_shift (Low-value category grows from 10% to 45%)
            elif name == "pure_mix_shift.csv":
                # Books/Apparel as low-value
                if week > 10:
                    weight = min(0.45, 0.10 + (week - 10) * (0.35 / 50))
                    if random.random() < weight:
                        r["category"] = "Books"
                        r["amount"] = round(r["amount"] * 0.3, 2)
                        
            # 4. systemic_shift (Weeks 30-40, all drop)
            elif name == "systemic_shift.csv":
                if 30 <= week <= 40:
                    r["amount"] = round(r["amount"] * 0.6, 2)
                    
            # 5. measurement_artefact (Week 35+, 40% missing amounts, volume halves, West->Western)
            elif name == "measurement_artefact.csv":
                if week >= 35:
                    if random.random() < 0.5: # halve volume roughly
                        continue
                    if random.random() < 0.4:
                        r["amount"] = ""
                    if r["region"] == "West":
                        r["region"] = "Western"
                        
            # 6. two_confounded_causes (Region North drop + mix shift)
            elif name == "two_confounded_causes.csv":
                if 25 <= week <= 35:
                    if r["region"] == "North":
                        r["amount"] = round(r["amount"] * 0.7, 2)
                    if random.random() < 0.3:
                        r["category"] = "Books"
                        r["amount"] = round(r["amount"] * 0.4, 2)
                        
            # 7. simpsons_paradox (Overall falls, every region rises via volume skew)
            elif name == "simpsons_paradox.csv":
                if 15 <= week <= 45:
                    if r["region"] == "North":
                        r["amount"] = round(r["amount"] * 1.5, 2)
                    else:
                        r["amount"] = round(r["amount"] * 0.4, 2)
                        
            # 8. upward_spike (Weeks 20-30, East spikes up)
            elif name == "upward_spike.csv":
                if 20 <= week <= 30 and r["region"] == "East":
                    r["amount"] = round(r["amount"] * 2.2, 2)
                    
            # 9. seasonal_only (Dec peak, weekend dips)
            elif name == "seasonal_only.csv":
                d_obj = datetime.strptime(r["order_date"], "%Y-%m-%d")
                if d_obj.month == 12:
                    r["amount"] = round(r["amount"] * 1.8, 2)
                if d_obj.weekday() >= 5:
                    continue # weekend dip via volume reduction
                    
            # 10. gradual_drift (Slow steady decline over 30 weeks)
            elif name == "gradual_drift.csv":
                if 10 <= week <= 40:
                    factor = 1.0 - (0.35 * ((week - 10) / 30.0))
                    r["amount"] = round(r["amount"] * factor, 2)
                    
            # 11. messy_format (Handled during write: delimiter=;, BOM, mixed dates, quotes)
            elif name == "messy_format.csv":
                if 20 <= week <= 28 and r["region"] == "North":
                    r["amount"] = round(r["amount"] * 0.55, 2)
                    r["customer_feedback"] = '"Delayed, damaged, and missing items!"'
                if random.random() < 0.03:
                    r["region"] = ""
                    
            # 12. thin_segments (40 distinct regions)
            elif name == "thin_segments.csv":
                r["region"] = f"Region_{random.randint(1, 40)}"

            rows.append(r)
    return rows

def write_files():
    scenarios = [
        "pure_noise.csv", "localised_event.csv", "pure_mix_shift.csv",
        "systemic_shift.csv", "measurement_artefact.csv", "two_confounded_causes.csv",
        "simpsons_paradox.csv", "upward_spike.csv", "seasonal_only.csv",
        "gradual_drift.csv", "messy_format.csv", "thin_segments.csv"
    ]
    
    for s in scenarios:
        print(f"Generating {s}...")
        data = generate_scenario(s)
        
        if s == "messy_format.csv":
            # Hostile formatting: semicolon delimiter, UTF-8 BOM, mixed date formats
            with open(s, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0].keys()), delimiter=";")
                writer.writeheader()
                for row in data:
                    d_obj = datetime.strptime(row["order_date"], "%Y-%m-%d")
                    fmt_choice = random.choice([1, 2, 3])
                    if fmt_choice == 1:
                        row["order_date"] = d_obj.strftime("%Y-%m-%d")
                    elif fmt_choice == 2:
                        row["order_date"] = d_obj.strftime("%d/%m/%Y")
                    else:
                        row["order_date"] = d_obj.strftime("%d %b %Y")
                    writer.writerow(row)
        else:
            with open(s, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                writer.writeheader()
                writer.writerows(data)

if __name__ == "__main__":
    write_files()
    print("All 12 adversarial datasets generated successfully.")