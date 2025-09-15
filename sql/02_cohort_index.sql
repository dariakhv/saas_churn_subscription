-- Postgres SQL: Compute cohort index (age in months) and monthly activity flags
-- Assumptions:
-- - subscriptions(customer_id, start_date, end_date)
-- - A customer is considered active in month m if any portion of an active subscription overlaps month m
-- - For simplicity, we expand months using generate_series between start_date and COALESCE(end_date, CURRENT_DATE)

SELECT
    s.customer_id,
    DATE_TRUNC('month', i.invoice_date) AS invoice_month,
    c.cohort_month,
    (
        EXTRACT(YEAR FROM i.invoice_date) * 12 + EXTRACT(MONTH FROM i.invoice_date)
        - (EXTRACT(YEAR FROM c.cohort_month) * 12 + EXTRACT(MONTH FROM c.cohort_month))
        + 1
    ) AS cohort_index
FROM invoices i
JOIN subscriptions s
    ON i.subscription_id = s.subscription_id
JOIN (
    SELECT 
        s.customer_id, 
        MIN(DATE_TRUNC('month', i.invoice_date)) AS cohort_month
    FROM invoices i
    JOIN subscriptions s
        ON i.subscription_id = s.subscription_id
    GROUP BY s.customer_id
) c 
    ON s.customer_id = c.customer_id;
