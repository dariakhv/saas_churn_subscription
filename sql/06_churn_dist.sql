-- Postgres SQL: This query returns mean and median dustributions of support calls vs churn
SELECT 
    subscription_type,
    churn,
    COUNT(*) AS total_customers,
    AVG(support_calls) AS avg_support_calls,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY support_calls) AS median_support_calls
FROM customers
GROUP BY subscription_type, churn
ORDER BY subscription_type, churn;