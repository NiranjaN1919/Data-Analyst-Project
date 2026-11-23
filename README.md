# Data Analytics Final Projects

Three end-to-end projects, each with data cleaning, analysis, visualisation, a written report and a dashboard.

| Project | Question | Tools | Key finding |
|---|---|---|---|
| 1. Sales Analysis Dashboard | Where do sales, profit and customers come from? | Power BI, SQL, Python | Interactive dashboard with KPIs, trends, category/region breakdown and customer tiers |
| 2. Customer Segmentation & Prediction | Which customer groups exist, and can we predict repeat purchases? | Python (pandas, scikit-learn, seaborn), SQL | 4 stable segments (silhouette 0.31, bootstrap ARI 0.92). Repeat-purchase models perform at chance because the customer IDs are unreliable; purchase value is predictable from category |
| 3. Financial Data Analysis & Forecasting | What are the trends in S&P 500 stocks, and can prices be forecast? | Python (pandas, seaborn, custom ARIMA/Holt-Winters), SQL, Power BI | Market +17% a year with three short-lived falls. Prices are not forecastable month to month; volatility is |

## What is in each project
- `01_Data`: cleaned data and cleaning log
- `02_Charts`: report charts
- `03_Models` or results files: model metrics
- `04_Dashboard`: dashboard (HTML, Power BI guide)
- `05_Report`: report (Word and PDF)
- Scripts: Python, SQL and report builders

## Highlights
- Every model is tested on data it did not see, against a naive baseline.
- Findings that did not work are reported as they are, with the reason (for example the customer ID collision test in Project 2, and survivorship bias in Project 3).
- SQL queries were run and checked against the Python results.
