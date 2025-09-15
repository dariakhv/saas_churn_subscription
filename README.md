# SaaS Customer Retention & Churn Analysis  

This project analyzes fictional customer retention and churn patterns, showcasing end-to-end data analytics skills from **data management with PostgreSQL**, to **SQL-based cohort analysis**, **Python-based visualization**, and **predictive modeling with XGBoost**.  

## Report  
Business case study and report can be found [HERE](file:///Users/daria/Documents/saas_churn_subscription/report/retention_report.html).

## Project Overview  
- **Business Problem:** High early churn in SaaS subscriptions threatens long-term revenue growth.  
- **KPIs Tracked:**  
  - Retention Rate (% of active users retained since signup)  
  - Churn Rate (% of users lost per period)  
- **Approach:**  
  1. Load data into **PostgreSQL (pgAdmin)** for storage & cohort queries  
  2. Analyze retention patterns with **SQL + Python (pandas, matplotlib, seaborn)**  
  3. Model churn risk using **XGBoost** for proactive customer targeting  
  4. Present findings in a structured **Quarto Report (HTML)**  

## Key Results  
- **Retention curves** show biggest churn occurs in the **first 1–3 months**.  
- **Support calls** strongly correlate with churn → friction in support operations.  
- Predictive churn model enables **early identification of at-risk users**.  