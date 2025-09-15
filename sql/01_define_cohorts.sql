-- Postgres SQL: Define acquisition cohorts (by month of first subscription start)
-- Assumptions:
-- - Source table: subscriptions(customer_id, start_date, end_date, is_active)
-- - Dates are of type date or timestamp; casted to date where needed
-- - Adjust table/column names if your schema differs

-- 1) Base: first known subscription start per customer
WITH first_subscription AS (
  SELECT
    s.customer_id,
    MIN(s.start_date::date) AS first_start_date
  FROM subscriptions s
  GROUP BY s.customer_id
),
-- 2) Acquisition cohort as month (date truncated to month start)
cohorts AS (
  SELECT
    f.customer_id,
    date_trunc('month', f.first_start_date)::date AS acquisition_month
  FROM first_subscription f
)
-- 3) Persist/return cohort mapping
SELECT *
FROM cohorts
ORDER BY acquisition_month, customer_id;
