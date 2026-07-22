-- =============================================================================
-- Uniqlo Sydney CBD — Division Performance Queries
-- =============================================================================

-- ---- 1. Division annual leaderboard (revenue + GP) ----------------------------
SELECT
    division_code,
    division_name,
    department,
    strftime('%Y', date)           AS year,
    ROUND(SUM(sales_amt), 0)       AS total_sales,
    ROUND(SUM(gp_amt), 0)          AS total_gp,
    ROUND(AVG(gp_ratio) * 100, 2)  AS avg_gp_pct,
    RANK() OVER (
        PARTITION BY strftime('%Y', date)
        ORDER BY SUM(sales_amt) DESC
    ) AS revenue_rank,
    RANK() OVER (
        PARTITION BY strftime('%Y', date)
        ORDER BY AVG(gp_ratio) DESC
    ) AS gp_pct_rank
FROM division_daily
GROUP BY division_code, year
ORDER BY year, revenue_rank;

-- ---- 2. Division revenue mix-shift (% of total revenue by month) ---------------
WITH monthly_div AS (
    SELECT
        strftime('%Y-%m', date) AS year_month,
        division_code,
        division_name,
        department,
        SUM(sales_amt) AS monthly_sales
    FROM division_daily
    GROUP BY year_month, division_code
),
monthly_totals AS (
    SELECT year_month, SUM(monthly_sales) AS total_monthly
    FROM monthly_div
    GROUP BY year_month
)
SELECT
    m.year_month,
    m.division_code,
    m.division_name,
    m.department,
    m.monthly_sales,
    ROUND(m.monthly_sales * 100.0 / t.total_monthly, 2) AS revenue_pct
FROM monthly_div m
JOIN monthly_totals t ON m.year_month = t.year_month
ORDER BY m.year_month, revenue_pct DESC;

-- ---- 3. YoY revenue change per division (2024 → 2025) -------------------------
WITH yearly AS (
    SELECT
        division_code,
        division_name,
        department,
        strftime('%Y', date) AS year,
        SUM(sales_amt) AS annual_sales
    FROM division_daily
    GROUP BY division_code, year
)
SELECT
    a.division_code,
    a.division_name,
    a.department,
    a.annual_sales  AS sales_2024,
    b.annual_sales  AS sales_2025,
    ROUND((b.annual_sales - a.annual_sales) * 100.0 / a.annual_sales, 2) AS yoy_pct
FROM yearly a
JOIN yearly b
    ON a.division_code = b.division_code
    AND a.year = '2024' AND b.year = '2025'
ORDER BY yoy_pct DESC;

-- ---- 4. Loungewear YoY growth highlight (brief: strong YoY trend) ---------------
SELECT
    strftime('%Y-%m', date) AS year_month,
    division_code,
    division_name,
    SUM(sales_amt) AS monthly_sales,
    LAG(SUM(sales_amt), 12) OVER (
        PARTITION BY division_code
        ORDER BY strftime('%Y-%m', date)
    ) AS same_month_ly,
    ROUND(
        (SUM(sales_amt) - LAG(SUM(sales_amt), 12) OVER (
            PARTITION BY division_code
            ORDER BY strftime('%Y-%m', date)
        )) * 100.0
        / NULLIF(LAG(SUM(sales_amt), 12) OVER (
            PARTITION BY division_code
            ORDER BY strftime('%Y-%m', date)
        ), 0),
    2) AS yoy_pct
FROM division_daily
WHERE division_code IN (28, 38)  -- Loungewear divisions
GROUP BY year_month, division_code
ORDER BY division_code, year_month;

-- ---- 5. Outerwear cold-snap responsiveness (Div 21 & 31) -----------------------
SELECT
    CASE
        WHEN ds.temperature_index < 12 THEN 'Very Cold (<12°C)'
        WHEN ds.temperature_index < 15 THEN 'Cold (12–15°C)'
        WHEN ds.temperature_index < 20 THEN 'Mild (15–20°C)'
        ELSE 'Warm (>20°C)'
    END AS temp_band,
    dd.division_code,
    dd.division_name,
    COUNT(DISTINCT ds.date)         AS num_days,
    ROUND(AVG(dd.sales_amt), 0)     AS avg_daily_sales,
    ROUND(MAX(dd.sales_amt), 0)     AS peak_daily_sales
FROM division_daily dd
JOIN daily_sales ds ON dd.date = ds.date
WHERE dd.division_code IN (21, 31)
GROUP BY temp_band, dd.division_code
ORDER BY MIN(ds.temperature_index), dd.division_code;

-- ---- 6. Women's Dress (Div 29) — low GP% awareness ------------------------------
SELECT
    strftime('%Y-%m', date)        AS year_month,
    ROUND(SUM(sales_amt), 0)       AS monthly_sales,
    ROUND(AVG(gp_ratio) * 100, 2)  AS avg_gp_pct,
    ROUND(SUM(gp_amt), 0)          AS monthly_gp,
    ROUND(
        SUM(sales_amt) * 100.0 / (
            SELECT SUM(sales_amt)
            FROM division_daily dd2
            WHERE strftime('%Y-%m', dd2.date) = strftime('%Y-%m', dd.date)
        ), 2
    ) AS pct_of_store_monthly
FROM division_daily dd
WHERE division_code = 29
GROUP BY year_month
ORDER BY year_month;

-- ---- 7. Department-level monthly contribution ----------------------------------
SELECT
    strftime('%Y-%m', date)        AS year_month,
    department,
    ROUND(SUM(sales_amt), 0)       AS dept_sales,
    ROUND(AVG(gp_ratio) * 100, 2)  AS avg_gp_pct,
    ROUND(
        SUM(sales_amt) * 100.0 /
        SUM(SUM(sales_amt)) OVER (PARTITION BY strftime('%Y-%m', date)),
    2) AS dept_pct
FROM division_daily
GROUP BY year_month, department
ORDER BY year_month, dept_sales DESC;

-- ---- 8. Top SKUs by revenue (join products) ------------------------------------
SELECT
    ti.sku_id,
    p.product_name,
    p.division_name,
    p.department,
    p.price_tier,
    p.unit_price,
    COUNT(ti.transaction_id)        AS num_transactions,
    SUM(ti.quantity)                AS units_sold,
    ROUND(SUM(ti.discounted_price * ti.quantity), 0) AS total_revenue,
    ROUND(SUM(ti.gross_profit_amt), 0) AS total_gp
FROM transaction_items ti
JOIN products p ON ti.sku_id = p.sku_id
GROUP BY ti.sku_id, p.product_name, p.division_name, p.department, p.price_tier, p.unit_price
ORDER BY total_revenue DESC
LIMIT 30;
