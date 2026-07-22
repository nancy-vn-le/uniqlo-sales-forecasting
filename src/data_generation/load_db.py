"""
Load all CSV files from data/raw/ into SQLite data/uniqlo.db.
Applies indexes after bulk load for performance.
Run after generate_data.py.
"""

from pathlib import Path
import pandas as pd
from sqlalchemy import text
from src.utils.db_connect import get_engine

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"

TABLE_FILES = [
    ("products",           "products.csv"),
    ("customers",          "customers.csv"),
    ("daily_sales",        "daily_sales.csv"),
    ("division_daily",     "division_daily.csv"),
    ("transactions",       "transactions.csv"),
    ("transaction_items",  "transaction_items.csv"),
    ("inventory",          "inventory.csv"),
]

POST_LOAD_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_products_division ON products(division_code)",
    "CREATE INDEX IF NOT EXISTS idx_products_dept ON products(department)",
    "CREATE INDEX IF NOT EXISTS idx_customers_type ON customers(customer_type)",
    "CREATE INDEX IF NOT EXISTS idx_customers_nationality ON customers(nationality)",
    "CREATE INDEX IF NOT EXISTS idx_daily_sales_event ON daily_sales(event_flag)",
    "CREATE INDEX IF NOT EXISTS idx_daily_sales_dow ON daily_sales(day_of_week)",
    "CREATE INDEX IF NOT EXISTS idx_div_daily_div ON division_daily(division_code)",
    "CREATE INDEX IF NOT EXISTS idx_div_daily_dept ON division_daily(department)",
    "CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(date)",
    "CREATE INDEX IF NOT EXISTS idx_txn_customer ON transactions(customer_id)",
    "CREATE INDEX IF NOT EXISTS idx_txn_type ON transactions(customer_type)",
    "CREATE INDEX IF NOT EXISTS idx_items_txn ON transaction_items(transaction_id)",
    "CREATE INDEX IF NOT EXISTS idx_items_sku ON transaction_items(sku_id)",
    "CREATE INDEX IF NOT EXISTS idx_items_div ON transaction_items(division_code)",
    "CREATE INDEX IF NOT EXISTS idx_inventory_sku ON inventory(sku_id)",
    "CREATE INDEX IF NOT EXISTS idx_inventory_oos ON inventory(oos_flag)",
]


def load_all() -> None:
    engine = get_engine()
    print("=" * 60)
    print("Loading CSVs into SQLite: data/uniqlo.db")
    print("=" * 60)

    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA synchronous=NORMAL"))

    for table_name, filename in TABLE_FILES:
        path = RAW_DIR / filename
        if not path.exists():
            print(f"  SKIP  {filename} — file not found")
            continue

        print(f"  Loading {filename} ...", end=" ", flush=True)
        df = pd.read_csv(path)

        # Use executemany (not multi-row INSERT) to avoid SQLite variable limit
        df.to_sql(table_name, engine, if_exists="replace", index=False,
                  chunksize=5_000)
        print(f"{len(df):,} rows loaded")

    print()
    print("Applying indexes ...")
    with engine.connect() as conn:
        for idx_sql in POST_LOAD_INDEXES:
            conn.execute(text(idx_sql))
        conn.commit()

    print()
    print("Verifying row counts:")
    with engine.connect() as conn:
        for table_name, _ in TABLE_FILES:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.scalar()
            print(f"  {table_name}: {count:,}")

    print()
    print("Done. Database ready at data/uniqlo.db")


if __name__ == "__main__":
    load_all()
