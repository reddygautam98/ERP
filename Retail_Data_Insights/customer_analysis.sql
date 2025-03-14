-- Top-Selling Products by Quantity
START TRANSACTION;
SELECT 
    product_name, 
    SUM(quantity_sold) AS Total_Quantity_Sold, 
    SUM(total_amount) AS Total_Revenue
FROM erp_sales
GROUP BY product_name
ORDER BY Total_Quantity_Sold DESC
LIMIT 10;
COMMIT;

-- Payment Method Distribution
START TRANSACTION;
SELECT 
    payment_method, 
    COUNT(transaction_id) AS Transaction_Count, 
    SUM(total_amount) AS Total_Spent
FROM erp_sales
GROUP BY payment_method
ORDER BY Total_Spent DESC;
COMMIT;

-- Stock Availability Impact on Sales
START TRANSACTION;
SELECT 
    stock_availability, 
    COUNT(transaction_id) AS Transaction_Count, 
    SUM(total_amount) AS Total_Revenue
FROM erp_sales
GROUP BY stock_availability;
COMMIT;

-- Monthly Revenue Trend
START TRANSACTION;
SELECT 
    DATE_FORMAT(date_of_sale, '%Y-%m') AS Month, 
    SUM(total_amount) AS Total_Revenue
FROM erp_sales
GROUP BY Month
ORDER BY Month ASC;
COMMIT;
