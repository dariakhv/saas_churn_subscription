-- Postgres SQL: Aggregate retention counts per cohort and cohort_index
WITH customer_cohorts AS (
    SELECT 
        s.customer_id,
        MIN(DATE_TRUNC('month', i.invoice_date)) AS cohort_month
    FROM invoices i
    JOIN subscriptions s 
        ON i.subscription_id = s.subscription_id
    GROUP BY s.customer_id
),
invoice_with_cohort AS (
    SELECT
        s.customer_id,
        DATE_TRUNC('month', i.invoice_date) AS invoice_month,
        cc.cohort_month
    FROM invoices i
    JOIN subscriptions s 
        ON i.subscription_id = s.subscription_id
    JOIN customer_cohorts cc 
        ON s.customer_id = cc.customer_id
),
cohort_indexed AS (
    SELECT
        customer_id,
        cohort_month,
        invoice_month,
        EXTRACT(MONTH FROM AGE(invoice_month, cohort_month)) + 1 AS cohort_index
    FROM invoice_with_cohort
)
SELECT
    cohort_month,
    cohort_index,
    COUNT(DISTINCT customer_id) AS num_customers
FROM cohort_indexed
GROUP BY cohort_month, cohort_index
ORDER BY cohort_month, cohort_index;