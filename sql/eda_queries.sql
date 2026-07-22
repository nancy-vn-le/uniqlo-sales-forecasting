-- =============================================================================
-- Uniqlo Sydney CBD — EDA Queries
-- =============================================================================

-- ---- 1. Daily sales summary ---------------------------------------------------
SELECT
    date,
    day_of_week,
    event_flag,
    temperature_index,
    actual_sales,
    target,
    achievement_pct,
    gross_profit_amt,
    gross_profit_ratio,
    num_customers,
    avg_spend
FROM daily_sales
ORDER BY date;

-- ---- 2. Weekly sales aggregation ---------------------------------------------
SELECT
    strftime('%Y-W%W', date)  AS year_week,
    MIN(date)                  AS week_start,
    SUM(actual_sales)          AS weekly_sales,
    SUM(gross_profit_amt)      AS weekly_gp,
    ROUND(AVG(gross_profit_ratio), 4) AS avg_gp_pct,
    SUM(num_customers)         AS total_customers,
    ROUND(AVG(achievement_pct), 4)    AS avg_achievement,
    COUNT(*)                   AS trading_days
FROM daily_sales
GROUP BY year_week
ORDER BY year_week;

-- ---- 3. Monthly sales summary ------------------------------------------------
SELECT
    strftime('%Y-%m', date) AS year_month,
    SUM(actual_sales)        AS monthly_sales,
    SUM(gross_profit_amt)    AS monthly_gp,
    ROUND(AVG(gross_profit_ratio), 4) AS avg_gp_pct,
    ROUND(AVG(actual_sales), 0)       AS avg_daily_sales,
    SUM(num_customers)                AS total_customers,
    COUNT(CASE WHEN achievement_pct >= 1.0 THEN 1 END) AS days_on_target,
    COUNT(*)                          AS trading_days,
    ROUND(
        1.0 * COUNT(CASE WHEN achievement_pct >= 1.0 THEN 1 END) / COUNT(*), 4
    ) AS target_hit_rate
FROM daily_sales
GROUP BY year_month
ORDER BY year_month;

-- ---- 4. Day-of-week performance profile --------------------------------------
SELECT
    day_of_week,
    COUNT(*)                           AS num_days,
    ROUND(AVG(actual_sales), 0)        AS avg_sales,
    ROUND(AVG(target), 0)              AS avg_target,
    ROUND(AVG(achievement_pct), 4)     AS avg_achievement,
    ROUND(MIN(actual_sales), 0)        AS min_sales,
    ROUND(MAX(actual_sales), 0)        AS max_sales,
    ROUND(AVG(num_customers), 0)       AS avg_customers,
    ROUND(AVG(avg_spend), 2)           AS avg_basket
FROM daily_sales
GROUP BY day_of_week
ORDER BY
    CASE day_of_week
        WHEN 'Monday' THEN 1 WHEN 'Tuesday' THEN 2 WHEN 'Wednesday' THEN 3
        WHEN 'Thursday' THEN 4 WHEN 'Friday' THEN 5
        WHEN 'Saturday' THEN 6 WHEN 'Sunday' THEN 7
    END;

-- ---- 5. Event flag impact ----------------------------------------------------
SELECT
    event_flag,
    COUNT(*)                        AS num_days,
    ROUND(AVG(actual_sales), 0)     AS avg_sales,
    ROUND(AVG(achievement_pct), 4)  AS avg_achievement,
    ROUND(AVG(num_customers), 0)    AS avg_customers,
    ROUND(AVG(avg_spend), 2)        AS avg_basket,
    ROUND(AVG(gross_profit_ratio), 4) AS avg_gp_pct
FROM daily_sales
GROUP BY event_flag
ORDER BY avg_sales DESC;

-- ---- 6. Top divisions by annual revenue (2024) --------------------------------
SELECT
    dd.division_code,
    dd.division_name,
    dd.department,
    ROUND(SUM(dd.sales_amt), 0)    AS total_sales,
    ROUND(SUM(dd.gp_amt), 0)       AS total_gp,
    ROUND(AVG(dd.gp_ratio), 4)     AS avg_gp_pct,
    ROUND(
        SUM(dd.sales_amt) * 100.0 /
        (SELECT SUM(sales_amt) FROM division_daily WHERE strftime('%Y', date) = '2024'),
        2
    ) AS pct_of_total
FROM division_daily dd
WHERE strftime('%Y', date) = '2024'
GROUP BY dd.division_code, dd.division_name, dd.department
ORDER BY total_sales DESC;

-- ---- 7. Division revenue share by month (2024) --------------------------------
SELECT
    strftime('%Y-%m', dd.date)  AS year_month,
    dd.division_code,
    dd.division_name,
    ROUND(SUM(dd.sales_amt), 0) AS monthly_sales
FROM division_daily dd
WHERE strftime('%Y', date) = '2024'
GROUP BY year_month, dd.division_code
ORDER BY year_month, monthly_sales DESC;

-- ---- 8. Seasonal demand heatmap: avg daily sales by month × division ----------
SELECT
    CAST(strftime('%m', date) AS INTEGER) AS month_num,
    division_code,
    division_name,
    ROUND(AVG(sales_amt), 0) AS avg_daily_sales
FROM division_daily
GROUP BY month_num, division_code
ORDER BY month_num, division_code;

-- ---- 9. Cold-snap effect on outerwear divisions --------------------------------
SELECT
    CASE
        WHEN ds.temperature_index < 12 THEN 'Very Cold (<12°C)'
        WHEN ds.temperature_index < 15 THEN 'Cold (12–15°C)'
        WHEN ds.temperature_index < 20 THEN 'Mild (15–20°C)'
        ELSE 'Warm (>20°C)'
    END AS temp_band,
    ROUND(AVG(CASE WHEN dd.division_code IN (21, 31) THEN dd.sales_amt END), 0) AS avg_outerwear_sales,
    ROUND(AVG(CASE WHEN dd.division_code IN (24, 34) THEN dd.sales_amt END), 0) AS avg_cutsewn_sales,
    COUNT(DISTINCT ds.date) AS num_days
FROM daily_sales ds
JOIN division_daily dd ON ds.date = dd.date
GROUP BY temp_band
ORDER BY MIN(ds.temperature_index);

-- ---- 10. Customer segment overview --------------------------------------------
SELECT
    customer_type,
    COUNT(*)                            AS num_customers,
    ROUND(AVG(avg_spend_per_visit), 2)  AS avg_spend,
    ROUND(AVG(total_visits), 1)         AS avg_visits,
    ROUND(AVG(total_spend), 2)          AS avg_total_spend,
    ROUND(SUM(total_spend), 0)          AS sum_total_spend
FROM customers
GROUP BY customer_type
ORDER BY sum_total_spend DESC;

-- ---- 11. YoY comparison by month (2024 vs 2025) --------------------------------
SELECT
    strftime('%m', date)             AS month_num,
    SUM(CASE WHEN strftime('%Y', date) = '2024' THEN actual_sales ELSE 0 END) AS sales_2024,
    SUM(CASE WHEN strftime('%Y', date) = '2025' THEN actual_sales ELSE 0 END) AS sales_2025,
    ROUND(
        (SUM(CASE WHEN strftime('%Y', date) = '2025' THEN actual_sales ELSE 0 END)
         - SUM(CASE WHEN strftime('%Y', date) = '2024' THEN actual_sales ELSE 0 END))
        * 100.0
        / NULLIF(SUM(CASE WHEN strftime('%Y', date) = '2024' THEN actual_sales ELSE 0 END), 0),
    2) AS yoy_pct_change
FROM daily_sales
GROUP BY month_num
ORDER BY month_num;

-- ---- 12. GP% by division ranked -----------------------------------------------
SELECT
    division_code,
    division_name,
    department,
    ROUND(AVG(gp_ratio) * 100, 2)  AS avg_gp_pct,
    ROUND(SUM(sales_amt), 0)        AS total_sales,
    ROUND(SUM(gp_amt), 0)           AS total_gp
FROM division_daily
GROUP BY division_code, division_name, department
ORDER BY avg_gp_pct DESC;
