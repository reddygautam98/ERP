-- Total Sales and Revenue
START TRANSACTION;
SELECT 
    SUM(total_amount) AS total_revenue, 
    COUNT(transaction_id) AS total_sales
FROM erp_sales;
COMMIT;

-- Sales by Category
START TRANSACTION;
SELECT 
    category, 
    SUM(quantity_sold) AS total_quantity, 
    SUM(total_amount) AS total_revenue
FROM erp_sales
GROUP BY category
ORDER BY total_revenue DESC;
COMMIT;

-- Top Selling Products
START TRANSACTION;
SELECT 
    product_name, 
    SUM(quantity_sold) AS total_quantity, 
    SUM(total_amount) AS total_revenue
FROM erp_sales
GROUP BY product_name
ORDER BY total_quantity DESC
LIMIT 10;
COMMIT;

-- Sales Trend by Month
START TRANSACTION;
SELECT 
    DATE_FORMAT(date_of_sale, '%Y-%m') AS month, 
    SUM(total_amount) AS total_revenue
FROM erp_sales
GROUP BY month
ORDER BY month ASC;
COMMIT;

-- Sales by Location
START TRANSACTION;
SELECT 
    location, 
    SUM(total_amount) AS total_revenue, 
    COUNT(transaction_id) AS total_sales
FROM erp_sales
GROUP BY location
ORDER BY total_revenue DESC;