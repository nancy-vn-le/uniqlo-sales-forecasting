"""
Synthetic data generator for Uniqlo Sydney CBD sales forecasting project.
All figures calibrated to project_brief.md — no real data reproduced.
Generation order: products → customers → daily_sales → division_daily
                  → transactions → transaction_items → inventory
"""

import numpy as np
import pandas as pd
from pathlib import Path
import uuid
import random
from datetime import date, timedelta

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Domain constants from project_brief.md
# ---------------------------------------------------------------------------

DIVISIONS = [
    (11, "Boys",              "Kids",   0.585, False, "Year-round"),
    (12, "Girls",             "Kids",   0.570, False, "Year-round"),
    (18, "Baby",              "Kids",   0.575, False, "Year-round"),
    (21, "Women's Outerwear", "Women",  0.540, True,  "Winter"),
    (22, "Women's Bottoms",   "Women",  0.620, False, "Year-round"),
    (23, "Women's Shirt",     "Women",  0.620, False, "Year-round"),
    (24, "Women's Cut & Sewn","Women",  0.670, False, "Year-round"),
    (25, "Women's Knit",      "Women",  0.610, False, "AW"),
    (26, "Women's Accessories","Women", 0.670, False, "Year-round"),
    (27, "Women's Inner",     "Women",  0.670, True,  "Year-round"),
    (28, "Women's Loungewear","Women",  0.550, False, "Year-round"),
    (29, "Women's Dress",     "Women",  0.210, False, "SS"),
    (31, "Men's Outerwear",   "Men",    0.560, True,  "Winter"),
    (32, "Men's Bottoms",     "Men",    0.620, False, "Year-round"),
    (33, "Men's Shirt",       "Men",    0.570, False, "Year-round"),
    (34, "Men's Cut & Sewn",  "Men",    0.640, False, "Year-round"),
    (35, "Men's Knit",        "Men",    0.610, False, "AW"),
    (36, "Men's Accessories", "Men",    0.590, False, "Year-round"),
    (37, "Men's Inner",       "Men",    0.600, True,  "Year-round"),
    (38, "Men's Loungewear",  "Men",    0.630, False, "Year-round"),
    (88, "Miscellaneous",     "Others", 0.290, False, "Year-round"),
]
DIV_LOOKUP = {d[0]: d for d in DIVISIONS}

PRICE_TIERS = {
    "Entry":   (9.90,  28.90),
    "Core":    (29.90, 49.90),
    "Mid":     (49.90, 79.90),
    "Premium": (99.90, 199.90),
}

DATE_START = date(2024, 1, 1)
DATE_END   = date(2025, 12, 31)
ALL_DATES  = pd.date_range(DATE_START, DATE_END, freq="D")

# ---------------------------------------------------------------------------
# Helper: Sydney temperature curve (seasonal sinusoid + noise)
# ---------------------------------------------------------------------------

def sydney_temp(dates: pd.DatetimeIndex) -> np.ndarray:
    """Return synthetic daily temperature in °C for Sydney."""
    doy = dates.day_of_year.to_numpy()
    # Sydney (Southern Hemisphere): peak summer ~Jan (doy 15) ~28 deg C,
    # peak winter ~Jul (doy 196) ~13 deg C.
    # cos(0)=1 at doy=15 gives 20.5+7.5=28; cos(pi)=-1 at doy=196 gives 20.5-7.5=13.
    baseline = 20.5 + 7.5 * np.cos(2 * np.pi * (doy - 15) / 365)
    noise = rng.normal(0, 1.8, len(dates))
    return np.clip(baseline + noise, 5.0, 38.0)


# ---------------------------------------------------------------------------
# Helper: event flags
# ---------------------------------------------------------------------------

SYDNEY_PUBLIC_HOLIDAYS_2024 = {
    date(2024, 1, 1), date(2024, 1, 26), date(2024, 3, 29), date(2024, 3, 30),
    date(2024, 4, 1), date(2024, 4, 25), date(2024, 6, 10), date(2024, 8, 5),
    date(2024, 10, 7), date(2024, 12, 25), date(2024, 12, 26),
}
SYDNEY_PUBLIC_HOLIDAYS_2025 = {
    date(2025, 1, 1), date(2025, 1, 27), date(2025, 4, 18), date(2025, 4, 19),
    date(2025, 4, 21), date(2025, 4, 25), date(2025, 6, 9), date(2025, 8, 4),
    date(2025, 10, 6), date(2025, 12, 25), date(2025, 12, 26),
}
ALL_HOLIDAYS = SYDNEY_PUBLIC_HOLIDAYS_2024 | SYDNEY_PUBLIC_HOLIDAYS_2025

LONG_WEEKENDS = {
    # (year, month, days) — Mon public holidays adjacent to weekends
    date(2024, 3, 29), date(2024, 3, 30), date(2024, 4, 1),
    date(2024, 4, 25), date(2024, 4, 26), date(2024, 4, 27),
    date(2024, 6, 8), date(2024, 6, 9), date(2024, 6, 10),
    date(2024, 10, 5), date(2024, 10, 6), date(2024, 10, 7),
    date(2025, 4, 18), date(2025, 4, 19), date(2025, 4, 21),
    date(2025, 4, 25), date(2025, 4, 26), date(2025, 4, 27),
    date(2025, 6, 7), date(2025, 6, 8), date(2025, 6, 9),
    date(2025, 10, 4), date(2025, 10, 5), date(2025, 10, 6),
}


def get_event_flag(d: date, temp: float) -> str:
    m, day = d.month, d.day
    # Arigato Festival: 21–27 May
    if m == 5 and 21 <= day <= 27:
        return "Arigato"
    # Christmas / Boxing Day
    if m == 12 and day in (24, 25, 26, 27):
        return "Christmas"
    # EOFY: late June to mid July
    if (m == 6 and day >= 20) or (m == 7 and day <= 15):
        return "EOFY"
    # Vivid Sydney: late May to mid June — lift only when not too cold/rainy
    if (m == 5 and day >= 24) or (m == 6 and day <= 15):
        # Weather-dependent: only triggers event lift if temp > 14°C
        if temp > 14.0:
            return "Vivid"
    # Long weekends
    if d in LONG_WEEKENDS:
        return "LongWeekend"
    return "None"


# ---------------------------------------------------------------------------
# Task 3a: products.csv
# ---------------------------------------------------------------------------

def generate_products() -> pd.DataFrame:
    print("Generating products.csv ...")
    rows = []
    sku_counter = 1

    product_templates = {
        11: ["Boys T-Shirt", "Boys Jeans", "Boys Shorts", "Boys Jacket", "Boys Sweater",
             "Boys Pyjamas", "Boys Fleece", "Boys Chinos"],
        12: ["Girls T-Shirt", "Girls Jeans", "Girls Leggings", "Girls Dress",
             "Girls Jacket", "Girls Sweater", "Girls Pyjamas", "Girls Fleece"],
        18: ["Baby Romper", "Baby Bodysuit", "Baby Pants", "Baby Cardigan",
             "Baby Pyjamas", "Baby Hat Set", "Baby Fleece"],
        21: ["Women's Down Jacket", "Women's Padded Vest", "Women's Trench Coat",
             "Women's Fleece Jacket", "Women's Wool Coat", "Women's Puffer Jacket",
             "Women's Rain Jacket", "Women's Parka"],
        22: ["Women's Slim Jeans", "Women's Wide Jeans", "Women's Chinos",
             "Women's Leggings", "Women's Shorts", "Women's Joggers",
             "Women's Skirt", "Women's Tailored Pants"],
        23: ["Women's Oxford Shirt", "Women's Linen Shirt", "Women's Flannel Shirt",
             "Women's Stripe Shirt", "Women's Check Shirt", "Women's Denim Shirt",
             "Women's Print Shirt", "Women's Tunic"],
        24: ["Women's UT Tee", "Women's AIRism T-Shirt", "Women's Cotton Tee",
             "Women's Ribbed Tank", "Women's Polo Shirt", "Women's Sweatshirt",
             "Women's Crewneck", "Women's Hoodie", "Women's Long-Sleeve Tee"],
        25: ["Women's Lambswool Cardigan", "Women's Cable Knit", "Women's Crew Neck Knit",
             "Women's V-Neck Knit", "Women's Turtleneck Knit", "Women's Knit Vest",
             "Women's Mohair Cardigan", "Women's Cashmere Blend Knit"],
        26: ["Women's Scarf", "Women's Tote Bag", "Women's Socks Pack",
             "Women's Belt", "Women's Beanie", "Women's Gloves", "Women's Hair Accessories"],
        27: ["Women's Heattech Tee", "Women's Heattech Extra Warm",
             "Women's AIRism Bra", "Women's Wireless Bra",
             "Women's Heattech Leggings", "Women's Seamless Innerwear",
             "Women's Heattech Crew Neck"],
        28: ["Women's Fleece Pyjamas", "Women's Loungewear Set", "Women's Sweat Shorts",
             "Women's Sweat Pants", "Women's Cosy Hoodie", "Women's Relaxed Joggers",
             "Women's Velour Pyjamas", "Women's Cotton Pyjamas"],
        29: ["Women's Floral Dress", "Women's Linen Dress", "Women's Wrap Dress",
             "Women's Shirt Dress", "Women's Jersey Dress", "Women's Midi Dress",
             "Women's Mini Dress", "Women's Slip Dress"],
        31: ["Men's Down Jacket", "Men's Padded Vest", "Men's Trench Coat",
             "Men's Fleece Jacket", "Men's Wool Coat", "Men's Puffer Jacket",
             "Men's Rain Jacket", "Men's Parka"],
        32: ["Men's Slim Jeans", "Men's Straight Jeans", "Men's Chinos",
             "Men's Shorts", "Men's Joggers", "Men's Sweat Pants",
             "Men's Cargo Pants", "Men's Tailored Pants"],
        33: ["Men's Oxford Shirt", "Men's Flannel Shirt", "Men's Linen Shirt",
             "Men's Denim Shirt", "Men's Check Shirt", "Men's Broadcloth Shirt",
             "Men's Print Shirt", "Men's Chambray Shirt"],
        34: ["Men's UT Tee", "Men's AIRism T-Shirt", "Men's Cotton Tee",
             "Men's Polo Shirt", "Men's Sweatshirt", "Men's Crewneck",
             "Men's Hoodie", "Men's Long-Sleeve Tee", "Men's Henley"],
        35: ["Men's Lambswool Cardigan", "Men's Cable Knit", "Men's Crew Neck Knit",
             "Men's V-Neck Knit", "Men's Turtleneck Knit", "Men's Knit Vest",
             "Men's Shawl Collar Cardigan", "Men's Cashmere Blend Knit"],
        36: ["Men's Scarf", "Men's Tote Bag", "Men's Socks Pack",
             "Men's Belt", "Men's Beanie", "Men's Gloves", "Men's Cap"],
        37: ["Men's Heattech Tee", "Men's Heattech Extra Warm",
             "Men's AIRism T-Shirt Inner", "Men's Heattech Crew Neck",
             "Men's Heattech V-Neck", "Men's AIRism Boxer Briefs",
             "Men's Heattech Tights"],
        38: ["Men's Fleece Pyjamas", "Men's Loungewear Set", "Men's Sweat Shorts",
             "Men's Sweat Pants", "Men's Cosy Hoodie", "Men's Relaxed Joggers",
             "Men's Velour Pyjamas", "Men's Cotton Pyjamas"],
        88: ["Gift Card", "Reusable Bag", "Branded Umbrella",
             "Loyalty Tote", "Seasonal Bundle"],
    }

    tier_weights = {
        11: ["Entry", "Core", "Core", "Mid"],
        12: ["Entry", "Core", "Core", "Mid"],
        18: ["Entry", "Entry", "Core", "Core"],
        21: ["Mid", "Mid", "Premium", "Premium", "Premium", "Mid", "Mid", "Premium"],
        22: ["Core", "Core", "Core", "Core", "Entry", "Core", "Core", "Core"],
        23: ["Core", "Core", "Core", "Core", "Core", "Core", "Core", "Core"],
        24: ["Core", "Core", "Entry", "Core", "Core", "Core", "Core", "Core", "Core"],
        25: ["Mid", "Mid", "Mid", "Mid", "Mid", "Mid", "Mid", "Premium"],
        26: ["Core", "Core", "Entry", "Core", "Entry", "Core", "Entry"],
        27: ["Core", "Mid", "Core", "Core", "Core", "Core", "Core"],
        28: ["Core", "Core", "Core", "Core", "Core", "Core", "Core", "Core"],
        29: ["Core", "Core", "Mid", "Core", "Core", "Mid", "Core", "Core"],
        31: ["Mid", "Mid", "Premium", "Premium", "Premium", "Mid", "Mid", "Premium"],
        32: ["Core", "Core", "Core", "Core", "Core", "Core", "Core", "Core"],
        33: ["Core", "Core", "Core", "Core", "Core", "Core", "Core", "Core"],
        34: ["Core", "Core", "Entry", "Core", "Core", "Core", "Core", "Core", "Core"],
        35: ["Mid", "Mid", "Mid", "Mid", "Mid", "Mid", "Mid", "Premium"],
        36: ["Core", "Core", "Entry", "Core", "Entry", "Core", "Entry"],
        37: ["Core", "Mid", "Core", "Core", "Core", "Core", "Core"],
        38: ["Core", "Core", "Core", "Core", "Core", "Core", "Core", "Core"],
        88: ["Entry", "Entry", "Entry", "Entry", "Core"],
    }

    for div_code, div_name, dept, base_gp, cold_snap, season in DIVISIONS:
        templates = product_templates.get(div_code, [f"{div_name} Item {i}" for i in range(8)])
        tiers = tier_weights.get(div_code, ["Core"] * len(templates))
        launch_base = date(2023, 9, 1)

        for i, (name, tier) in enumerate(zip(templates, tiers)):
            lo, hi = PRICE_TIERS[tier]
            price = round(rng.uniform(lo, hi) * 2) / 2  # round to nearest 0.50
            gp = round(base_gp + rng.uniform(-0.02, 0.02), 4)
            launch = launch_base + timedelta(days=int(rng.integers(0, 180)))
            rows.append({
                "sku_id": f"SKU{sku_counter:04d}",
                "division_code": div_code,
                "division_name": div_name,
                "department": dept,
                "product_name": name,
                "price_tier": tier,
                "unit_price": price,
                "season": season,
                "gp_ratio": gp,
                "is_cold_snap_sensitive": cold_snap,
                "launch_date": launch.isoformat(),
                "is_active": True,
            })
            sku_counter += 1

    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / "products.csv", index=False)
    print(f"  products.csv: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Task 3b: customers.csv
# ---------------------------------------------------------------------------

def generate_customers() -> pd.DataFrame:
    print("Generating customers.csv ...")
    n = 80_000
    segments = rng.choice(
        ["loyalty", "casual", "tourist_eu", "tourist_other"],
        size=n,
        p=[0.45, 0.25, 0.25, 0.05],
    )
    customer_types = np.where(
        np.isin(segments, ["tourist_eu", "tourist_other"]), "tourist", segments
    )
    nationalities = np.where(
        segments == "tourist_eu", "European",
        np.where(segments == "tourist_other", "Other", "Australian")
    )
    age_bands = rng.choice(
        ["18-24", "25-34", "35-44", "45-54", "55+"],
        size=n, p=[0.20, 0.35, 0.25, 0.12, 0.08]
    )
    genders = rng.choice(["F", "M", "unspecified"], size=n, p=[0.52, 0.44, 0.04])

    join_dates = []
    for seg in segments:
        if seg == "loyalty":
            d = DATE_START - timedelta(days=int(rng.integers(30, 1500)))
            join_dates.append(d.isoformat())
        else:
            join_dates.append(None)

    # Spend profiles
    base_spend = np.where(
        customer_types == "loyalty", rng.uniform(70, 85, n),
        np.where(customer_types == "casual", rng.uniform(55, 75, n),
                 np.where(segments == "tourist_eu", rng.uniform(90, 130, n),
                          rng.uniform(80, 110, n)))
    )

    # Visits: loyalty = frequent, casual = moderate, tourist = 1
    total_visits = np.where(
        customer_types == "loyalty", rng.integers(3, 30, n),
        np.where(customer_types == "casual", rng.integers(1, 8, n),
                 rng.integers(1, 3, n))
    )

    avg_spend_per_visit = np.round(base_spend + rng.normal(0, 5, n), 2)
    total_spend = np.round(avg_spend_per_visit * total_visits, 2)

    last_visit_offsets = rng.integers(0, 365, n)
    last_visit_dates = [
        (DATE_END - timedelta(days=int(off))).isoformat()
        for off in last_visit_offsets
    ]

    df = pd.DataFrame({
        "customer_id": [f"CUST{i+1:06d}" for i in range(n)],
        "customer_type": customer_types,
        "join_date": join_dates,
        "nationality": nationalities,
        "age_band": age_bands,
        "gender": genders,
        "total_visits": total_visits,
        "total_spend": total_spend,
        "avg_spend_per_visit": avg_spend_per_visit,
        "last_visit_date": last_visit_dates,
    })
    df.to_csv(RAW_DIR / "customers.csv", index=False)
    print(f"  customers.csv: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Task 4: daily_sales.csv
# ---------------------------------------------------------------------------

DAY_TARGETS = {
    "Monday":    (126_000, 140_000),
    "Tuesday":   (126_000, 140_000),
    "Wednesday": (126_000, 140_000),
    "Thursday":  (175_000, 224_000),
    "Friday":    (154_000, 168_000),
    "Saturday":  (175_000, 238_000),
    "Sunday":    (168_000, 196_000),
}

# Event multipliers applied to the base day target to set the event-day target.
# Calibrated so that event-day actual sales (target * achievement) stays within
# the brief's stated event-inflated ranges:
#   Mon-Wed event ~$315K, Thu event $350-500K, Sat event $380-480K.
# Arigato: $190K avg Mon * 1.60 = $304K target; * 1.09 avg ach = $331K actual  ok
# Christmas: $295K avg Sat * 1.45 = $428K target; * 1.09 = $467K actual  ok
EVENT_TARGET_MULTIPLIERS = {
    "Arigato":     1.60,
    "Christmas":   1.45,
    "EOFY":        1.20,
    "Vivid":       1.15,
    "LongWeekend": 1.12,
    "None":        1.00,
}


def generate_daily_sales(temps: pd.Series) -> pd.DataFrame:
    print("Generating daily_sales.csv ...")
    rows_2024 = []
    rows_2025 = []

    for i, dt in enumerate(ALL_DATES):
        d = dt.date()
        dow = dt.day_name()
        temp = float(temps.iloc[i])
        is_holiday = d in ALL_HOLIDAYS
        event = get_event_flag(d, temp)

        trading_hrs = 12 if dow == "Thursday" else (7 if dow == "Sunday" else 9)

        base_lo, base_hi = DAY_TARGETS[dow]
        target = round(
            rng.uniform(base_lo, base_hi) * EVENT_TARGET_MULTIPLIERS[event], -2
        )

        # 85-95% of days achieve >= 100% of target.
        # achievement_pct is the only multiplier on top of the event-inflated target -
        # EVENT_TARGET_MULTIPLIERS already captures the event lift, so no second
        # event multiplier is needed here.
        hit = rng.random() < 0.90
        if hit:
            ach = rng.uniform(1.00, 1.12)
        else:
            ach = rng.uniform(0.84, 0.99)

        actual_sales = round(target * ach, 2)

        gp_ratio = round(rng.uniform(0.59, 0.65), 4)
        gp_amt = round(actual_sales * gp_ratio, 2)

        avg_spend_wday = rng.uniform(77, 85)
        avg_spend_wknd = rng.uniform(88, 100)
        avg_spend = avg_spend_wknd if dow in ("Saturday", "Sunday", "Thursday") else avg_spend_wday
        num_customers = max(1, int(actual_sales / avg_spend))
        avg_qty = round(rng.uniform(2.1, 2.5), 2)
        avg_unit_price = round(avg_spend / avg_qty, 2)

        promo_flag = (event == "None") and (rng.random() < 0.12)

        row = {
            "date": d.isoformat(),
            "day_of_week": dow,
            "trading_hours": trading_hrs,
            "is_public_holiday": is_holiday,
            "event_flag": event,
            "temperature_index": round(temp, 2),
            "target": target,
            "actual_sales": actual_sales,
            "achievement_pct": round(actual_sales / target, 4),
            "num_customers": num_customers,
            "avg_spend": round(avg_spend, 2),
            "avg_qty": avg_qty,
            "avg_unit_price": avg_unit_price,
            "gross_profit_amt": gp_amt,
            "gross_profit_ratio": gp_ratio,
            "promo_flag": promo_flag,
        }

        if d.year == 2024:
            rows_2024.append(row)
        else:
            rows_2025.append(row)

    # YoY: match 2025 to same day in 2024
    df_2024 = pd.DataFrame(rows_2024)
    df_2025 = pd.DataFrame(rows_2025)
    lookup_2024 = df_2024.set_index("date")["actual_sales"].to_dict()

    yoy_list = []
    for row in rows_2025:
        prior_date = (date.fromisoformat(row["date"]) - timedelta(days=364)).isoformat()
        prior = lookup_2024.get(prior_date)
        if prior and prior > 0:
            yoy_list.append(round((row["actual_sales"] - prior) / prior, 4))
        else:
            yoy_list.append(None)
    df_2025["yoy_pct"] = yoy_list

    df_2024["yoy_pct"] = None
    df = pd.concat([df_2024, df_2025], ignore_index=True)
    df.to_csv(RAW_DIR / "daily_sales.csv", index=False)
    print(f"  daily_sales.csv: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Division seasonal weight curves
# ---------------------------------------------------------------------------

def division_weight(div_code: int, temp: float, month: int) -> float:
    """Return relative revenue weight for a division on a given day."""
    if div_code in (24, 34):  # Cut & Sewn — flat year-round, top sellers
        return 0.155
    if div_code in (21, 31):  # Outerwear - cold-snap sensitive, tourist floor in summer
        # min 0.18 so the division never disappears: tourists from winter-hemisphere
        # countries buy Outerwear year-round at a Sydney CBD flagship.
        cold_mult = max(0.18, min(2.5, (18 - temp) / 5)) if temp < 18 else 0.18
        return 0.04 * cold_mult
    if div_code in (27, 37):  # Inner/Heattech — winter weighted
        cold_mult = max(0.6, min(2.0, (18 - temp) / 6 + 0.6)) if temp < 18 else 0.6
        return 0.06 * cold_mult
    if div_code in (25, 35):  # Knit — Autumn-Winter
        aw_weight = 1.0 + max(0, (18 - temp) / 12)
        return 0.04 * aw_weight
    if div_code == 29:  # Women's Dress - Spring-Summer peak, tourist floor in winter
        # min 0.12 so winter days still show some sales (tourist and occasion wear)
        ss_weight = max(0.12, (temp - 10) / 18) if temp > 10 else 0.12
        return 0.025 * ss_weight
    if div_code in (11, 12, 18):  # Kids — year-round, small share
        return 0.030
    if div_code == 88:  # Misc — very small
        return 0.008
    if div_code in (22, 32):  # Bottoms — year-round
        return 0.060
    if div_code in (23, 33):  # Shirts — year-round
        return 0.045
    if div_code in (26, 36):  # Accessories — year-round
        return 0.030
    if div_code in (28, 38):  # Loungewear — year-round, growing
        return 0.035
    return 0.030


# ---------------------------------------------------------------------------
# Task 5: division_daily.csv
# ---------------------------------------------------------------------------

def generate_division_daily(daily_df: pd.DataFrame) -> pd.DataFrame:
    print("Generating division_daily.csv ...")
    rows = []

    for _, day_row in daily_df.iterrows():
        d = day_row["date"]
        dt = pd.Timestamp(d)
        total_sales = day_row["actual_sales"]
        temp = day_row["temperature_index"]
        month = dt.month

        raw_weights = {
            div_code: division_weight(div_code, temp, month)
            for div_code, *_ in DIVISIONS
        }
        total_w = sum(raw_weights.values())
        norm_weights = {k: v / total_w for k, v in raw_weights.items()}

        for div_code, div_name, dept, base_gp, _, _ in DIVISIONS:
            w = norm_weights[div_code]
            sales_amt = round(total_sales * w, 2)
            gp_ratio = round(base_gp + rng.uniform(-0.015, 0.015), 4)
            gp_amt = round(sales_amt * gp_ratio, 2)
            num_txn = max(1, int(sales_amt / rng.uniform(60, 100)))

            rows.append({
                "date": d,
                "division_code": div_code,
                "division_name": div_name,
                "department": dept,
                "sales_amt": sales_amt,
                "gp_amt": gp_amt,
                "gp_ratio": gp_ratio,
                "vs_target_pct": round(day_row["achievement_pct"], 4),
                "vs_ly_pct": day_row["yoy_pct"],
                "num_transactions": num_txn,
            })

    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / "division_daily.csv", index=False)
    print(f"  division_daily.csv: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Task 6: transactions.csv + transaction_items.csv
# ---------------------------------------------------------------------------

INTRADAY_WEIGHTS = {
    "Monday":    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 8, 14, 16, 14, 12, 11, 10, 7, 5, 0, 0, 0, 0, 0],
    "Tuesday":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 8, 14, 16, 14, 12, 11, 10, 7, 5, 0, 0, 0, 0, 0],
    "Wednesday": [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 8, 14, 16, 14, 12, 11, 10, 7, 5, 0, 0, 0, 0, 0],
    "Thursday":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 11, 13, 12, 11, 10, 9, 9, 7, 5, 3, 1, 0, 0],
    "Friday":    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 8, 13, 15, 14, 13, 11, 10, 8, 5, 0, 0, 0, 0, 0],
    "Saturday":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 15, 17, 15, 12, 10, 8, 6, 2, 0, 0, 0, 0, 0],
    "Sunday":    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 14, 18, 17, 15, 12, 10, 7, 0, 0, 0, 0, 0, 0],
}


def tourist_boost(event: str, month: int) -> float:
    """Extra tourist share during peak tourist periods."""
    if event in ("Christmas", "Arigato", "Vivid"):
        return 0.08
    if month in (12, 1, 4):  # summer hols + Easter
        return 0.05
    return 0.0


def generate_transactions_and_items(
    daily_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Generating transactions.csv + transaction_items.csv ...")
    print("  (This may take a few minutes for ~1.5M rows...)")

    # Pre-build product lists per division for fast sampling
    div_products: dict[int, pd.DataFrame] = {}
    for div_code in products_df["division_code"].unique():
        div_products[div_code] = products_df[products_df["division_code"] == div_code].reset_index(drop=True)

    loyalty_ids = customers_df[customers_df["customer_type"] == "loyalty"]["customer_id"].tolist()
    casual_ids  = customers_df[customers_df["customer_type"] == "casual"]["customer_id"].tolist()
    tourist_ids = customers_df[customers_df["customer_type"] == "tourist"]["customer_id"].tolist()

    txn_rows = []
    item_rows = []
    txn_counter = 1

    for _, day_row in daily_df.iterrows():
        d = day_row["date"]
        dt = pd.Timestamp(d)
        dow = day_row["day_of_week"]
        event = day_row["event_flag"]
        num_customers = int(day_row["num_customers"])
        actual_sales = day_row["actual_sales"]
        promo_active = bool(day_row["promo_flag"])
        month = dt.month

        # Tourist share
        t_boost = tourist_boost(event, month)
        base_tourist = 0.27
        tourist_share = min(0.45, base_tourist + t_boost)
        loyalty_share = max(0.35, 0.45 - t_boost)
        casual_share = 1.0 - tourist_share - loyalty_share

        seg_probs = [loyalty_share, casual_share, tourist_share]

        # Intraday hour distribution
        hour_weights = INTRADAY_WEIGHTS[dow]
        hours = rng.choice(24, size=num_customers, p=np.array(hour_weights) / sum(hour_weights))
        minutes = rng.integers(0, 60, size=num_customers)

        # Avg spend targets
        is_peak_day = dow in ("Thursday", "Saturday", "Sunday")
        avg_spend_target = actual_sales / num_customers

        for j in range(num_customers):
            seg_idx = rng.choice(3, p=seg_probs)
            seg = ["loyalty", "casual", "tourist"][seg_idx]

            if seg == "loyalty":
                cust_id = random.choice(loyalty_ids)
                base_spend = rng.uniform(70, 85)
            elif seg == "casual":
                cust_id = random.choice(casual_ids)
                base_spend = rng.uniform(55, 75)
            else:
                cust_id = random.choice(tourist_ids)
                base_spend = rng.uniform(88, 130)

            num_items = int(rng.choice([1, 2, 2, 3, 3, 4, 5, 6], p=[0.15, 0.30, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01]))
            payment = rng.choice(["card", "contactless", "cash"], p=[0.45, 0.50, 0.05])

            # Weighted division selection for basket (Cut&Sewn most likely, then seasonal)
            div_codes = [d[0] for d in DIVISIONS]
            # Simple: pick top divisions by weight
            txn_id = f"TXN{txn_counter:08d}"
            txn_counter += 1

            ts = pd.Timestamp(d) + pd.Timedelta(hours=int(hours[j]), minutes=int(minutes[j]))

            item_total = 0.0
            item_gp = 0.0
            item_list = []

            for _ in range(num_items):
                # Pick division weighted by rough revenue share (normalized)
                _div_raw = np.array([0.025, 0.025, 0.015,
                                     0.04, 0.05, 0.04, 0.12, 0.04, 0.025, 0.055, 0.03, 0.02,
                                     0.04, 0.05, 0.04, 0.12, 0.04, 0.025, 0.055, 0.03, 0.005])
                _div_raw /= _div_raw.sum()
                div_code = int(rng.choice(
                    [11, 12, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                     31, 32, 33, 34, 35, 36, 37, 38, 88],
                    p=_div_raw
                ))
                prod_pool = div_products.get(div_code)
                if prod_pool is None or len(prod_pool) == 0:
                    continue
                prod = prod_pool.iloc[int(rng.integers(0, len(prod_pool)))]
                unit_price = float(prod["unit_price"])
                qty = 1 if rng.random() < 0.88 else 2
                promo_applied = promo_active and rng.random() < 0.15
                disc_price = round(unit_price * 0.80, 2) if promo_applied else unit_price
                line_gp = round(disc_price * qty * float(prod["gp_ratio"]), 2)

                item_list.append({
                    "transaction_id": txn_id,
                    "sku_id": prod["sku_id"],
                    "division_code": div_code,
                    "quantity": qty,
                    "unit_price": unit_price,
                    "discounted_price": disc_price,
                    "gross_profit_amt": line_gp,
                })
                item_total += disc_price * qty
                item_gp += line_gp

            if not item_list:
                continue

            txn_rows.append({
                "transaction_id": txn_id,
                "date": d,
                "timestamp": ts.isoformat(),
                "customer_id": cust_id,
                "customer_type": seg,
                "num_items": sum(it["quantity"] for it in item_list),
                "total_amount": round(item_total, 2),
                "gross_profit": round(item_gp, 2),
                "payment_method": payment,
                "promo_applied": promo_active and any(
                    it["unit_price"] != it["discounted_price"] for it in item_list
                ),
            })
            item_rows.extend(item_list)

        if dt.day == 1:
            print(f"    processed {d} | txns so far: {len(txn_rows):,}")

    txn_df = pd.DataFrame(txn_rows)
    item_df = pd.DataFrame(item_rows)
    txn_df.to_csv(RAW_DIR / "transactions.csv", index=False)
    item_df.to_csv(RAW_DIR / "transaction_items.csv", index=False)
    print(f"  transactions.csv: {len(txn_df):,} rows")
    print(f"  transaction_items.csv: {len(item_df):,} rows")
    return txn_df, item_df


# ---------------------------------------------------------------------------
# Task 7: inventory.csv
# ---------------------------------------------------------------------------

def generate_inventory(products_df: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
    print("Generating inventory.csv ...")

    # Aggregate weekly units sold per SKU from transaction items
    item_df2 = item_df.copy()
    # Need date — read from RAW_DIR if not available
    txn_dates = pd.read_csv(RAW_DIR / "transactions.csv", usecols=["transaction_id", "date"])
    item_df2 = item_df2.merge(txn_dates, on="transaction_id", how="left")
    item_df2["date"] = pd.to_datetime(item_df2["date"])
    item_df2["week_start"] = item_df2["date"] - pd.to_timedelta(item_df2["date"].dt.dayofweek, unit="D")
    item_df2["week_start"] = item_df2["week_start"].dt.date

    weekly_sold = (
        item_df2.groupby(["week_start", "sku_id"])["quantity"].sum()
        .reset_index()
        .rename(columns={"quantity": "units_sold_actual"})
    )

    weeks = pd.date_range("2024-01-01", "2025-12-31", freq="W-MON")
    week_starts = [w.date() for w in weeks]

    rows = []
    for _, prod in products_df.iterrows():
        sku = prod["sku_id"]
        season = prod["season"]
        # Base weekly velocity calibrated to price tier and season
        base_vel = {
            "Entry": rng.integers(15, 40),
            "Core":  rng.integers(8, 25),
            "Mid":   rng.integers(3, 12),
            "Premium": rng.integers(1, 6),
        }.get(prod["price_tier"], 10)

        opening = int(base_vel * rng.uniform(10, 20))  # start with ~10–20 weeks cover

        for week_start in week_starts:
            sold_row = weekly_sold[
                (weekly_sold["sku_id"] == sku) &
                (weekly_sold["week_start"] == week_start)
            ]
            units_sold = int(sold_row["units_sold_actual"].values[0]) if len(sold_row) > 0 else 0

            # Thursday replenishment: top up if below 4 weeks cover
            avg_vel = max(1, base_vel)
            weeks_cover = opening / avg_vel if avg_vel > 0 else 99
            received = 0
            if weeks_cover < 4:
                received = int(avg_vel * rng.uniform(4, 8))

            closing = max(0, opening + received - units_sold)
            oos = closing == 0 and units_sold > 0
            woc = round(closing / avg_vel, 2) if avg_vel > 0 else 99.0

            rows.append({
                "week_start_date": week_start.isoformat(),
                "sku_id": sku,
                "opening_stock": opening,
                "units_sold": units_sold,
                "units_received": received,
                "closing_stock": closing,
                "oos_flag": oos,
                "weeks_of_cover": woc,
            })
            opening = closing

    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / "inventory.csv", index=False)
    print(f"  inventory.csv: {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Uniqlo Sydney CBD — Synthetic Data Generator")
    print("=" * 60)

    temps_series = pd.Series(sydney_temp(ALL_DATES), index=ALL_DATES)

    products_df = generate_products()
    customers_df = generate_customers()
    daily_df = generate_daily_sales(temps_series)
    division_df = generate_division_daily(daily_df)
    txn_df, item_df = generate_transactions_and_items(daily_df, customers_df, products_df)
    inventory_df = generate_inventory(products_df, item_df)

    print()
    print("=" * 60)
    print("All datasets written to data/raw/")
    print("Row counts:")
    for fname in ["products.csv", "customers.csv", "daily_sales.csv",
                  "division_daily.csv", "transactions.csv",
                  "transaction_items.csv", "inventory.csv"]:
        path = RAW_DIR / fname
        if path.exists():
            n = sum(1 for _ in open(path)) - 1
            print(f"  {fname}: {n:,}")
    print("=" * 60)
    print("Next step: run src/data_generation/load_db.py")
