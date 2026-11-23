-- =====================================================================
-- Project 2 - Customer Segmentation: SQL data extraction
-- Tables : orders          (01_Data/ecommerce_orders_cleaned.csv)
--          segments        (01_Data/customer_segments.csv)  [only for Q4]
-- Dialect: SQLite (tested).  For SQL Server use DATEDIFF(day, MAX(order_date), '2025-09-12')
--          and for MySQL use DATEDIFF('2025-09-12', MAX(order_date)).
-- Reference date = day after the last order in the data (2025-09-12).
-- =====================================================================

-- Q1. One row per customer: RFM, order value, discount, returns, delivery
--     (the feature table used for clustering, same as customer_features.csv)
SELECT customer_id,
       CAST(julianday('2025-09-12') - julianday(MAX(order_date)) AS INTEGER) AS recency_days,
       COUNT(*)                                        AS frequency,
       ROUND(SUM(total_amount), 2)                     AS monetary,
       ROUND(SUM(total_amount) / COUNT(*), 2)          AS avg_order_value,
       ROUND(AVG(discount), 4)                         AS avg_discount,
       ROUND(AVG(is_returned), 4)                      AS return_rate,
       ROUND(AVG(delivery_time_days), 2)               AS avg_delivery_days,
       COUNT(DISTINCT category)                        AS n_categories,
       CAST(julianday(MAX(order_date)) - julianday(MIN(order_date)) AS INTEGER) AS tenure_days,
       ROUND(SUM(profit_margin), 2)                    AS total_profit
FROM orders
GROUP BY customer_id;

-- Q2. Favourite category per customer (largest share of spend)
WITH cat_spend AS (
    SELECT customer_id, category, SUM(total_amount) AS spend,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY SUM(total_amount) DESC) AS rn
    FROM orders
    GROUP BY customer_id, category
)
SELECT customer_id, category AS favourite_category, ROUND(spend, 2) AS category_spend
FROM cat_spend
WHERE rn = 1;

-- Q3. Customers who bought again: orders in the 180 days after a snapshot date
--     (the target behind the 'buy again' model, snapshot = 2025-03-12)
SELECT h.customer_id,
       COUNT(DISTINCT h.order_id)                      AS orders_before,
       CASE WHEN f.customer_id IS NULL THEN 0 ELSE 1 END AS bought_again_180d,
       COALESCE(f.future_spend, 0)                     AS future_spend_180d
FROM orders h
LEFT JOIN (SELECT customer_id, SUM(total_amount) AS future_spend
           FROM orders
           WHERE order_date >= '2025-03-12' AND order_date < date('2025-03-12', '+180 day')
           GROUP BY customer_id) f ON f.customer_id = h.customer_id
WHERE h.order_date < '2025-03-12'
GROUP BY h.customer_id;

-- Q4. Segment summary (after Python has written the segment label)
SELECT s.segment,
       COUNT(*)                                        AS customers,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM segments), 1)   AS pct_customers,
       ROUND(SUM(s.monetary), 2)                       AS revenue,
       ROUND(100.0 * SUM(s.monetary) / (SELECT SUM(monetary) FROM segments), 1) AS pct_revenue,
       ROUND(AVG(s.avg_order_value), 2)                AS avg_order_value,
       ROUND(AVG(s.frequency), 2)                      AS avg_orders,
       ROUND(AVG(s.avg_discount), 3)                   AS avg_discount
FROM segments s
GROUP BY s.segment
ORDER BY revenue DESC;
