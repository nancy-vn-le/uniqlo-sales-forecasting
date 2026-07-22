-- =============================================================================
-- Uniqlo Sydney CBD — Database Schema
-- SQLite-compatible; syntax is PostgreSQL-compatible (no SQLite-specific types)
-- =============================================================================

-- Products catalogue
CREATE TABLE IF NOT EXISTS products (
    sku_id                TEXT PRIMARY KEY,
    division_code         INTEGER NOT NULL,
    division_name         TEXT NOT NULL,
    department            TEXT NOT NULL,
    product_name          TEXT NOT NULL,
    price_tier            TEXT NOT NULL,
    unit_price            REAL NOT NULL,
    season                TEXT NOT NULL,
    gp_ratio              REAL NOT NULL,
    is_cold_snap_sensitive INTEGER NOT NULL DEFAULT 0,  -- BOOLEAN stored as 0/1
    launch_date           TEXT NOT NULL,
    is_active             INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_products_division ON products(division_code);
CREATE INDEX IF NOT EXISTS idx_products_dept ON products(department);

-- Customers
CREATE TABLE IF NOT EXISTS customers (
    customer_id          TEXT PRIMARY KEY,
    customer_type        TEXT NOT NULL,          -- loyalty / casual / tourist
    join_date            TEXT,                   -- NULL for non-loyalty
    nationality          TEXT NOT NULL,
    age_band             TEXT NOT NULL,
    gender               TEXT NOT NULL,
    total_visits         INTEGER NOT NULL DEFAULT 0,
    total_spend          REAL NOT NULL DEFAULT 0,
    avg_spend_per_visit  REAL NOT NULL DEFAULT 0,
    last_visit_date      TEXT
);

CREATE INDEX IF NOT EXISTS idx_customers_type ON customers(customer_type);
CREATE INDEX IF NOT EXISTS idx_customers_nationality ON customers(nationality);

-- Daily store-level sales (730 rows — 2024-01-01 to 2025-12-31)
CREATE TABLE IF NOT EXISTS daily_sales (
    date                 TEXT PRIMARY KEY,
    day_of_week          TEXT NOT NULL,
    trading_hours        INTEGER NOT NULL,
    is_public_holiday    INTEGER NOT NULL DEFAULT 0,
    event_flag           TEXT NOT NULL DEFAULT 'None',
    temperature_index    REAL NOT NULL,
    target               REAL NOT NULL,
    actual_sales         REAL NOT NULL,
    achievement_pct      REAL NOT NULL,
    num_customers        INTEGER NOT NULL,
    avg_spend            REAL NOT NULL,
    avg_qty              REAL NOT NULL,
    avg_unit_price       REAL NOT NULL,
    gross_profit_amt     REAL NOT NULL,
    gross_profit_ratio   REAL NOT NULL,
    yoy_pct              REAL,                   -- NULL for 2024 rows
    promo_flag           INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_daily_sales_event ON daily_sales(event_flag);
CREATE INDEX IF NOT EXISTS idx_daily_sales_dow ON daily_sales(day_of_week);

-- Division-level daily sales (14,600 rows — 20 divisions × 730 days)
-- Primary table for forecasting model
CREATE TABLE IF NOT EXISTS division_daily (
    date             TEXT NOT NULL,
    division_code    INTEGER NOT NULL,
    division_name    TEXT NOT NULL,
    department       TEXT NOT NULL,
    sales_amt        REAL NOT NULL,
    gp_amt           REAL NOT NULL,
    gp_ratio         REAL NOT NULL,
    vs_target_pct    REAL,
    vs_ly_pct        REAL,
    num_transactions INTEGER NOT NULL,
    PRIMARY KEY (date, division_code)
);

CREATE INDEX IF NOT EXISTS idx_div_daily_div ON division_daily(division_code);
CREATE INDEX IF NOT EXISTS idx_div_daily_dept ON division_daily(department);

-- Transactions (~1.5M rows)
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id  TEXT PRIMARY KEY,
    date            TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    customer_id     TEXT NOT NULL,
    customer_type   TEXT NOT NULL,
    num_items       INTEGER NOT NULL,
    total_amount    REAL NOT NULL,
    gross_profit    REAL NOT NULL,
    payment_method  TEXT NOT NULL,
    promo_applied   INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_txn_customer ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_txn_type ON transactions(customer_type);

-- Transaction items (~3.3M rows)
CREATE TABLE IF NOT EXISTS transaction_items (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id   TEXT NOT NULL,
    sku_id           TEXT NOT NULL,
    division_code    INTEGER NOT NULL,
    quantity         INTEGER NOT NULL DEFAULT 1,
    unit_price       REAL NOT NULL,
    discounted_price REAL NOT NULL,
    gross_profit_amt REAL NOT NULL,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id),
    FOREIGN KEY (sku_id) REFERENCES products(sku_id)
);

CREATE INDEX IF NOT EXISTS idx_items_txn ON transaction_items(transaction_id);
CREATE INDEX IF NOT EXISTS idx_items_sku ON transaction_items(sku_id);
CREATE INDEX IF NOT EXISTS idx_items_div ON transaction_items(division_code);

-- Weekly inventory per SKU (~20,800 rows)
CREATE TABLE IF NOT EXISTS inventory (
    week_start_date TEXT NOT NULL,
    sku_id          TEXT NOT NULL,
    opening_stock   INTEGER NOT NULL,
    units_sold      INTEGER NOT NULL DEFAULT 0,
    units_received  INTEGER NOT NULL DEFAULT 0,
    closing_stock   INTEGER NOT NULL,
    oos_flag        INTEGER NOT NULL DEFAULT 0,
    weeks_of_cover  REAL NOT NULL,
    PRIMARY KEY (week_start_date, sku_id),
    FOREIGN KEY (sku_id) REFERENCES products(sku_id)
);

CREATE INDEX IF NOT EXISTS idx_inventory_sku ON inventory(sku_id);
CREATE INDEX IF NOT EXISTS idx_inventory_oos ON inventory(oos_flag);

-- Forecast log (populated during Layer 2/3 development)
CREATE TABLE IF NOT EXISTS forecast_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_date   TEXT NOT NULL,
    target_date     TEXT NOT NULL,
    division_code   INTEGER NOT NULL,
    predicted_sales REAL NOT NULL,
    lower_bound     REAL,
    upper_bound     REAL,
    actual_sales    REAL,
    error_pct       REAL,
    context_flags   TEXT
);

CREATE INDEX IF NOT EXISTS idx_flog_target ON forecast_log(target_date);
CREATE INDEX IF NOT EXISTS idx_flog_div ON forecast_log(division_code);
