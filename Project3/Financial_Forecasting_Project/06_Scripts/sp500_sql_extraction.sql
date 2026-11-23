-- =====================================================================
-- Project 3 - Financial Data Analysis: SQL data extraction (SQLite, tested)
-- Table   : prices(date TEXT 'YYYY-MM-DD', ticker, open, high, low, close, adj_close, volume)
--           (= the cleaned daily file; load the raw CSV with  .mode csv / .import  or pandas.to_sql)
-- SQL Server: replace julianday()/strftime() with DATEDIFF / FORMAT; MySQL: DATE_FORMAT().
-- =====================================================================

-- Q1. Data-quality check: rows, tickers, date range, zero-volume rows, impossible prices
SELECT COUNT(*)                                   AS n_rows,
       COUNT(DISTINCT ticker)                     AS n_tickers,
       MIN(date)                                  AS first_date,
       MAX(date)                                  AS last_date,
       SUM(volume = 0)                            AS zero_volume_rows,
       SUM(high < low OR close <= 0)              AS bad_price_rows
FROM prices;

-- Q2. Daily return per stock (adjusted close, previous trading day via LAG)
SELECT date, ticker, adj_close,
       adj_close / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS daily_return
FROM prices
ORDER BY ticker, date;

-- Q3. Month-end price, monthly return and traded value per stock (feeds the monthly dashboard table)
WITH m AS (
    SELECT ticker,
           strftime('%Y-%m', date)                          AS month,
           MAX(date)                                        AS last_day,
           SUM(close * volume) / 1e6                        AS traded_value_musd
    FROM prices
    GROUP BY ticker, strftime('%Y-%m', date)
)
SELECT m.ticker, m.month, p.adj_close,
       p.adj_close / LAG(p.adj_close) OVER (PARTITION BY m.ticker ORDER BY m.month) - 1 AS monthly_return,
       m.traded_value_musd
FROM m JOIN prices p ON p.ticker = m.ticker AND p.date = m.last_day
ORDER BY m.ticker, m.month;

-- Q4. 50-day and 200-day moving averages for one stock (window frame = current row and the 49 / 199 before it)
SELECT date, adj_close,
       AVG(adj_close) OVER (ORDER BY date ROWS BETWEEN 49  PRECEDING AND CURRENT ROW) AS sma_50,
       AVG(adj_close) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS sma_200
FROM prices
WHERE ticker = 'NVDA'
ORDER BY date;

-- Q5. Performance ranking between two dates (change the dates; this is the dashboard's "top and bottom performers")
WITH a AS (SELECT ticker, adj_close AS p0 FROM prices WHERE date = '2025-11-14'),
     b AS (SELECT ticker, adj_close AS p1 FROM prices WHERE date = '2026-11-16')
SELECT a.ticker, ROUND(100.0 * (b.p1 / a.p0 - 1), 1) AS return_pct
FROM a JOIN b USING (ticker)
ORDER BY return_pct DESC
LIMIT 10;

-- Q6. Most traded stocks over the last 12 months (average daily traded value, USD million)
SELECT ticker, ROUND(AVG(close * volume) / 1e6, 0) AS avg_daily_traded_musd
FROM prices
WHERE date > date('2026-11-16', '-12 months')
GROUP BY ticker
ORDER BY avg_daily_traded_musd DESC
LIMIT 10;

-- Q7. Equal-weight market return per day (average of all stock returns) - used to build the market index
WITH r AS (
    SELECT date,
           adj_close / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS ret
    FROM prices
)
SELECT date, AVG(ret) AS market_return, COUNT(ret) AS stocks
FROM r
WHERE ret IS NOT NULL
GROUP BY date
ORDER BY date;
