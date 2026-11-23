-- =====================================================================
-- Sales Analysis Dashboard - SQL data-extraction queries
-- Table: sales  (load 01_Data/ecommerce_sales_cleaned.csv into it)
-- Dialect: SQLite (tested). For SQL Server / MySQL swap strftime(...) for
-- FORMAT(order_date,'yyyy-MM') / DATE_FORMAT(order_date,'%Y-%m').
-- =====================================================================

-- Q1. Headline KPIs
SELECT COUNT(*)                          AS total_orders,
       COUNT(DISTINCT customer_id)       AS unique_customers,
       ROUND(SUM(total_amount), 2)       AS total_sales,
       ROUND(AVG(total_amount), 2)       AS avg_sales_per_order,
       ROUND(SUM(profit_margin), 2)      AS total_profit,
       ROUND(100.0 * AVG(is_returned), 2) AS return_rate_pct
FROM sales;

-- Q2. Sales by region
SELECT region,
       ROUND(SUM(total_amount), 2)  AS total_sales,
       ROUND(AVG(total_amount), 2)  AS avg_sales,
       COUNT(*)                     AS total_orders
FROM sales
GROUP BY region
ORDER BY total_sales DESC;

-- Q3. Sales by product category
SELECT category,
       ROUND(SUM(total_amount), 2)  AS total_sales,
       ROUND(AVG(total_amount), 2)  AS avg_sales,
       COUNT(*)                     AS total_orders
FROM sales
GROUP BY category
ORDER BY total_sales DESC;

-- Q4. Sales by month (time period)
SELECT strftime('%Y-%m', order_date) AS year_month,
       ROUND(SUM(total_amount), 2)   AS total_sales,
       ROUND(AVG(total_amount), 2)   AS avg_sales,
       COUNT(*)                      AS total_orders
FROM sales
GROUP BY year_month
ORDER BY year_month;

-- Q5. Sales by quarter
SELECT year_quarter,
       ROUND(SUM(total_amount), 2)  AS total_sales,
       ROUND(AVG(total_amount), 2)  AS avg_sales,
       COUNT(*)                     AS total_orders
FROM sales
GROUP BY year_quarter
ORDER BY year_quarter;

-- Q6. Region x category x month (feeds the dashboard filters)
SELECT region, category, strftime('%Y-%m', order_date) AS year_month,
       ROUND(SUM(total_amount), 2) AS total_sales,
       ROUND(AVG(total_amount), 2) AS avg_sales,
       COUNT(*)                    AS total_orders
FROM sales
GROUP BY region, category, year_month;

-- Q7. Top 10 products by sales (product_id is not unique across categories)
SELECT product_id, category,
       ROUND(SUM(total_amount), 2) AS total_sales,
       SUM(quantity)               AS units,
       ROUND(SUM(profit_margin), 2) AS profit
FROM sales
GROUP BY product_id, category
ORDER BY total_sales DESC
LIMIT 10;

-- Q8. Customer segmentation by total spend
--     Cut-offs = 50th / 80th / 95th percentile of customer spend.
WITH customer_spend AS (
    SELECT customer_id, SUM(total_amount) AS total_spend
    FROM sales
    GROUP BY customer_id
)
SELECT CASE WHEN total_spend >  726.73 THEN 'Platinum'
            WHEN total_spend >  257.77 THEN 'Gold'
            WHEN total_spend >   67.42 THEN 'Silver'
            ELSE 'Bronze' END                    AS segment,
       COUNT(*)                                  AS customers,
       ROUND(SUM(total_spend), 2)                AS total_spend,
       ROUND(100.0 * SUM(total_spend) / (SELECT SUM(total_amount) FROM sales), 1) AS pct_of_sales
FROM customer_spend
GROUP BY segment
ORDER BY total_spend;

-- Q9. Return rate and profit by category
SELECT category,
       ROUND(100.0 * AVG(is_returned), 2)               AS return_rate_pct,
       ROUND(SUM(profit_margin), 2)                      AS total_profit,
       ROUND(100.0 * SUM(profit_margin) / SUM(total_amount), 1) AS profit_pct_of_sales
FROM sales
GROUP BY category
ORDER BY profit_pct_of_sales DESC;
