-- Postgres SQL: This query returns retention % by subscription type; pivoting to a matrix can be done in BI tools or with crosstab
WITH customer_cohorts AS (
    SELECT 
        s.customer_id,
        MIN(DATE_TRUNC('month', i.invoice_date)) AS cohort_month
    FROM invoices i
    JOIN subscriptions s ON i.subscription_id = s.subscription_id
    GROUP BY s.customer_id
),
invoices_with_cohort AS (
    SELECT
        s.customer_id,
        s.subscription_type,
        DATE_TRUNC('month', i.invoice_date) AS invoice_month,
        c.cohort_month,
        EXTRACT(YEAR FROM i.invoice_date) * 12 + EXTRACT(MONTH FROM i.invoice_date) -
        (EXTRACT(YEAR FROM c.cohort_month) * 12 + EXTRACT(MONTH FROM c.cohort_month)) + 1
        AS cohort_index
    FROM invoices i
    JOIN subscriptions s ON i.subscription_id = s.subscription_id
    JOIN customer_cohorts c ON s.customer_id = c.customer_id
),
cohort_counts AS (
    SELECT
        subscription_type,
        cohort_month,
        cohort_index,
        COUNT(DISTINCT customer_id) AS num_customers
    FROM invoices_with_cohort
    GROUP BY subscription_type, cohort_month, cohort_index
),
cohort_sizes AS (
    SELECT
        subscription_type,
        cohort_month,
        COUNT(DISTINCT customer_id) AS cohort_size
    FROM invoices_with_cohort
    WHERE cohort_index = 1
    GROUP BY subscription_type, cohort_month
)
SELECT
    cc.subscription_type,
    cc.cohort_month,
    cc.cohort_index,
    cc.num_customers,
    cs.cohort_size,
    ROUND(cc.num_customers::decimal / cs.cohort_size, 3) AS retention_rate
FROM cohort_counts cc
JOIN cohort_sizes cs 
  ON cc.subscription_type = cs.subscription_type 
 AND cc.cohort_month = cs.cohort_month
ORDER BY subscription_type, cohort_month, cohort_index;
