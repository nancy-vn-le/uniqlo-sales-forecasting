-- =============================================================================
-- Uniqlo Sydney CBD — Window Function Queries
-- =============================================================================

-- ---- 1. 7-day rolling average of daily sales ----------------------------------
SELECT
    date,
    actual_sales,
    ROUND(
        AVG(actual_sales) OVER (
            ORDER BY date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 0
    ) AS rolling_7d_avg,
    ROUND(
        AVG(actual_sales) OVER (
            ORDER BY date
            ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
        ), 0
    ) AS rolling_28d_avg
FROM daily_sales
ORDER BY date;

-- ---- 2. Running cumulative sales and GP by year --------------------------------
SELECT
    date,
    strftime('%Y', date) AS year,
    actual_sales,
    gross_profit_amt,
    SUM(actual_sales) OVER (
        PARTITION BY strftime('%Y', date)
        ORDER BY date
        ROWS UNBOUNDED PRECEDING
    ) AS ytd_sales,
    SUM(gross_profit_amt) OVER (
        PARTITION BY strftime('%Y', date)
        ORDER BY date
        ROWS UNBOUNDED PRECEDING
    ) AS ytd_gp
FROM daily_sales
ORDER BY date;

-- ---- 3. Division rank by weekly revenue ----------------------------------------
WITH weekly_div AS (
    SELECT
        strftime('%Y-W%W', date) AS year_week,
        division_code,
        division_name,
        SUM(sales_amt) AS weekly_sales
    FROM division_daily
    GROUP BY year_week, division_code
)
SELECT
    year_week,
    division_code,
    division_name,
    weekly_sales,
    RANK() OVER (
        PARTITION BY year_week
        ORDER BY weekly_sales DESC
    ) AS weekly_rank
FROM weekly_div
ORDER BY year_week, weekly_rank;

-- ---- 4. Division rolling 4-week revenue (smoothed trend) -----------------------
WITH weekly_div AS (
    SELECT
        strftime('%Y-W%W', date) AS year_week,
        MIN(date)                AS week_start,
        division_code,
        division_name,
        SUM(sales_amt)           AS weekly_sales
    FROM division_daily
    GROUP BY year_week, division_code
)
SELECT
    year_week,
    week_start,
    division_code,
    division_name,
    weekly_sales,
    ROUND(
        AVG(weekly_sales) OVER (
            PARTITION BY division_code
            ORDER BY year_week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ), 0
    ) AS rolling_4w_avg
FROM weekly_div
ORDER BY division_code, year_week;

-- ---- 5. Day-over-day sales change and % change ----------------------------------
SELECT
    date,
    day_of_week,
    actual_sales,
    LAG(actual_sales, 1) OVER (ORDER BY date)  AS prev_day_sales,
    actual_sales - LAG(actual_sales, 1) OVER (ORDER BY date) AS dod_change,
    ROUND(
        (actual_sales - LAG(actual_sales, 1) OVER (ORDER BY date))
        * 100.0
        / NULLIF(LAG(actual_sales, 1) OVER (ORDER BY date), 0),
    2) AS dod_pct_change
FROM daily_sales
ORDER BY date;

-- ---- 6. Week-over-week comparison per division ----------------------------------
WITH weekly AS (
    SELECT
        strftime('%Y-W%W', date) AS year_week,
        division_code,
        division_name,
        SUM(sales_amt) AS weekly_sales
    FROM division_daily
    GROUP BY year_week, division_code
)
SELECT
    year_week,
    division_code,
    division_name,
    weekly_sales,
    LAG(weekly_sales, 1) OVER (
        PARTITION BY division_code ORDER BY year_week
    ) AS prev_week_sales,
    ROUND(
        (weekly_sales - LAG(weekly_sales, 1) OVER (
            PARTITION BY division_code ORDER BY year_week
        )) * 100.0
        / NULLIF(LAG(weekly_sales, 1) OVER (
            PARTITION BY division_code ORDER BY year_week
        ), 0),
    2) AS wow_pct
FROM weekly
ORDER BY division_code, year_week;

-- ---- 7. Cumulative division revenue share (how fast divisions accumulate value) -
WITH monthly AS (
    SELECT
        strftime('%Y-%m', date) AS year_month,
        division_code,
        division_name,
        SUM(sales_amt) AS monthly_sales
    FROM division_daily
    GROUP BY year_month, division_code
),
totals AS (
    SELECT year_month, SUM(monthly_sales) AS month_total
    FROM monthly GROUP BY year_month
)
SELECT
    m.year_month,
    m.division_code,
    m.division_name,
    m.monthly_sales,
    ROUND(m.monthly_sales * 100.0 / t.month_total, 2) AS pct_of_month,
    ROUND(
        SUM(m.monthly_sales * 100.0 / t.month_total) OVER (
            PARTITION BY m.year_month
            ORDER BY m.monthly_sales DESC
            ROWS UNBOUNDED PRECEDING
        ), 2
    ) AS cumulative_pct
FROM monthly m
JOIN totals t ON m.year_month = t.year_month
ORDER BY m.year_month, m.monthly_sales DESC;

-- ---- 8. Sales percentile bands per division across all days --------------------
SELECT
    division_code,
    division_name,
    ROUND(MIN(sales_amt), 0)  AS p0_min,
    ROUND(
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY sales_amt), 0
    ) AS p25,
    ROUND(
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY sales_amt), 0
    ) AS p50_median,
    ROUND(
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY sales_amt), 0
    ) AS p75,
    ROUND(MAX(sales_amt), 0)  AS p100_max
FROM division_daily
GROUP BY division_code, division_name
ORDER BY p50_median DESC;

-- Note: PERCENTILE_CONT is PostgreSQL syntax. In SQLite use:
-- SELECT division_code, sales_amt FROM division_daily ORDER BY division_code, sales_amt
-- then compute in Python/pandas. The query above documents intent for the PostgreSQL-compatible schema.

-- ---- 9. Rolling 7-day customer count and avg spend trend -----------------------
SELECT
    date,
    num_customers,
    avg_spend,
    ROUND(AVG(num_customers) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 0) AS rolling_7d_customers,
    ROUND(AVG(avg_spend) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2)     AS rolling_7d_avg_spend
FROM daily_sales
ORDER BY date;

-- ---- 10. Division GP rank within department by year ----------------------------
SELECT
    strftime('%Y', date)    AS year,
    division_code,
    division_name,
    department,
    ROUND(SUM(sales_amt), 0) AS total_sales,
    ROUND(AVG(gp_ratio) * 100, 2) AS avg_gp_pct,
    RANK() OVER (
        PARTITION BY strftime('%Y', date), department
        ORDER BY AVG(gp_ratio) DESC
    ) AS gp_rank_in_dept
FROM division_daily
GROUP BY year, division_code, division_name, department
ORDER BY year, department, gp_rank_in_dept;
