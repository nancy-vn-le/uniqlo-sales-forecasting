-- =============================================================================
-- Uniqlo Sydney CBD — RFM Segmentation
-- Recency / Frequency / Monetary scoring from transactions + customers tables
-- =============================================================================

-- ---- 1. Base RFM metrics per customer ------------------------------------------
WITH rfm_base AS (
    SELECT
        t.customer_id,
        c.customer_type,
        c.nationality,
        c.age_band,
        -- Recency: days since last purchase (lower = more recent)
        CAST(
            julianday((SELECT MAX(date) FROM daily_sales)) -
            julianday(MAX(t.date))
        AS INTEGER) AS recency_days,
        -- Frequency: number of transactions
        COUNT(t.transaction_id) AS frequency,
        -- Monetary: total spend
        ROUND(SUM(t.total_amount), 2) AS monetary
    FROM transactions t
    JOIN customers c ON t.customer_id = c.customer_id
    GROUP BY t.customer_id, c.customer_type, c.nationality, c.age_band
),
-- ---- 2. RFM score assignment (1–4 scale, higher = better) ----------------------
rfm_scored AS (
    SELECT
        customer_id,
        customer_type,
        nationality,
        age_band,
        recency_days,
        frequency,
        monetary,
        -- Recency score: lower days = higher score
        NTILE(4) OVER (ORDER BY recency_days DESC)  AS r_score,
        -- Frequency score: higher frequency = higher score
        NTILE(4) OVER (ORDER BY frequency ASC)       AS f_score,
        -- Monetary score: higher spend = higher score
        NTILE(4) OVER (ORDER BY monetary ASC)        AS m_score
    FROM rfm_base
),
-- ---- 3. Combined RFM score and segment label -----------------------------------
rfm_labelled AS (
    SELECT
        *,
        r_score + f_score + m_score AS rfm_total,
        CASE
            WHEN r_score = 4 AND f_score >= 3 AND m_score >= 3 THEN 'Champions'
            WHEN r_score >= 3 AND f_score >= 3                  THEN 'Loyal Customers'
            WHEN r_score = 4 AND f_score <= 2                   THEN 'Recent Customers'
            WHEN r_score >= 3 AND f_score <= 2 AND m_score >= 3 THEN 'Potential Loyalists'
            WHEN r_score <= 2 AND f_score >= 3 AND m_score >= 3 THEN 'At Risk'
            WHEN r_score = 1 AND f_score >= 3                   THEN 'Cant Lose Them'
            WHEN r_score <= 2 AND m_score >= 3                  THEN 'Hibernating High Value'
            ELSE 'Low Value'
        END AS rfm_segment
    FROM rfm_scored
)
SELECT *
FROM rfm_labelled
ORDER BY rfm_total DESC;

-- ---- 4. Segment summary statistics ---------------------------------------------
WITH rfm_base AS (
    SELECT
        t.customer_id,
        c.customer_type,
        CAST(
            julianday((SELECT MAX(date) FROM daily_sales)) -
            julianday(MAX(t.date))
        AS INTEGER) AS recency_days,
        COUNT(t.transaction_id) AS frequency,
        ROUND(SUM(t.total_amount), 2) AS monetary
    FROM transactions t
    JOIN customers c ON t.customer_id = c.customer_id
    GROUP BY t.customer_id, c.customer_type
),
rfm_scored AS (
    SELECT *,
        NTILE(4) OVER (ORDER BY recency_days DESC) AS r_score,
        NTILE(4) OVER (ORDER BY frequency ASC)      AS f_score,
        NTILE(4) OVER (ORDER BY monetary ASC)        AS m_score
    FROM rfm_base
),
rfm_labelled AS (
    SELECT *,
        CASE
            WHEN r_score = 4 AND f_score >= 3 AND m_score >= 3 THEN 'Champions'
            WHEN r_score >= 3 AND f_score >= 3                  THEN 'Loyal Customers'
            WHEN r_score = 4 AND f_score <= 2                   THEN 'Recent Customers'
            WHEN r_score >= 3 AND f_score <= 2 AND m_score >= 3 THEN 'Potential Loyalists'
            WHEN r_score <= 2 AND f_score >= 3 AND m_score >= 3 THEN 'At Risk'
            WHEN r_score = 1 AND f_score >= 3                   THEN 'Cant Lose Them'
            WHEN r_score <= 2 AND m_score >= 3                  THEN 'Hibernating High Value'
            ELSE 'Low Value'
        END AS rfm_segment
    FROM rfm_scored
)
SELECT
    rfm_segment,
    customer_type,
    COUNT(*)                         AS num_customers,
    ROUND(AVG(recency_days), 1)      AS avg_recency_days,
    ROUND(AVG(frequency), 1)         AS avg_frequency,
    ROUND(AVG(monetary), 2)          AS avg_monetary,
    ROUND(SUM(monetary), 0)          AS total_revenue
FROM rfm_labelled
GROUP BY rfm_segment, customer_type
ORDER BY rfm_segment, customer_type;

-- ---- 5. Do forecast drivers affect segments differently? -------------------------
-- Cross-tab: avg basket size per customer type during cold-snap vs normal days
SELECT
    CASE
        WHEN ds.temperature_index < 15 THEN 'Cold snap (<15°C)'
        ELSE 'Normal (>=15°C)'
    END AS temp_condition,
    t.customer_type,
    COUNT(t.transaction_id)         AS num_transactions,
    ROUND(AVG(t.total_amount), 2)   AS avg_basket,
    ROUND(AVG(t.num_items), 2)      AS avg_items
FROM transactions t
JOIN daily_sales ds ON t.date = ds.date
GROUP BY temp_condition, t.customer_type
ORDER BY temp_condition, t.customer_type;

-- ---- 6. Cross-tab: event weeks by customer segment ----------------------------
SELECT
    ds.event_flag,
    t.customer_type,
    COUNT(t.transaction_id)          AS num_transactions,
    ROUND(AVG(t.total_amount), 2)    AS avg_basket,
    ROUND(SUM(t.total_amount), 0)    AS total_spend
FROM transactions t
JOIN daily_sales ds ON t.date = ds.date
WHERE ds.event_flag != 'None'
GROUP BY ds.event_flag, t.customer_type
ORDER BY ds.event_flag, total_spend DESC;
