-- Total Sales and Transactions per Employee
START TRANSACTION;
SELECT 
    employee_name, 
    SUM(total_amount) AS total_sales, 
    COUNT(transaction_id) AS total_transactions, 
    AVG(total_amount) AS avg_transaction_value
FROM erp_sales
GROUP BY employee_name
ORDER BY total_sales DESC;
COMMIT;

-- Highest Selling Product per Employee
START TRANSACTION;
SELECT 
    employee_name, 
    product_name, 
    SUM(quantity_sold) AS total_quantity_sold
FROM erp_sales
GROUP BY employee_name, product_name
ORDER BY employee_name, total_quantity_sold DESC;
COMMIT;

-- Monthly Employee Revenue Trend
START TRANSACTION;
SELECT 
    employee_name, 
    DATE_FORMAT(date_of_sale, '%Y-%m') AS month, 
    SUM(total_amount) AS total_revenue
FROM erp_sales
GROUP BY employee_name, month
ORDER BY employee_name, month ASC;
COMMIT;

-- Employee Performance Ranking by Revenue
START TRANSACTION;
SELECT 
    employee_name, 
    SUM(total_amount) AS total_revenue, 
    RANK() OVER (ORDER BY SUM(total_amount) DESC) AS performance_rank
FROM erp_sales
GROUP BY employee_name;
COMMIT;