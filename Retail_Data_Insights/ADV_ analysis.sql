-- ==========================================
-- Sales Analysis SQL Script
-- ==========================================

-- Enable stored procedure creation
DELIMITER //

-- ==========================================
-- Table Creation
-- ==========================================

-- Main sales transaction table
CREATE TABLE sales_transactions (
  transaction_id VARCHAR(50) PRIMARY KEY,
  customer_name VARCHAR(100),
  employee_name VARCHAR(100),
  date_of_sale DATETIME,
  product_id VARCHAR(50),
  product_name VARCHAR(100),
  product_category VARCHAR(50),
  quantity_sold INT,
  price DECIMAL(10, 2),
  total_amount DECIMAL(10, 2),
  store_id VARCHAR(50),
  payment_method VARCHAR(50)
);

-- Dimension tables for a star schema
CREATE TABLE dim_customers (
  customer_id VARCHAR(50) PRIMARY KEY,
  customer_name VARCHAR(100),
  customer_email VARCHAR(100),
  customer_phone VARCHAR(20),
  customer_address VARCHAR(200),
  customer_city VARCHAR(50),
  customer_state VARCHAR(50),
  customer_zip VARCHAR(20),
  customer_segment VARCHAR(50)
);

CREATE TABLE dim_products (
  product_id VARCHAR(50) PRIMARY KEY,
  product_name VARCHAR(100),
  product_category VARCHAR(50),
  product_subcategory VARCHAR(50),
  product_cost DECIMAL(10, 2),
  product_price DECIMAL(10, 2)
);

CREATE TABLE dim_stores (
  store_id VARCHAR(50) PRIMARY KEY,
  store_name VARCHAR(100),
  store_address VARCHAR(200),
  store_city VARCHAR(50),
  store_state VARCHAR(50),
  store_zip VARCHAR(20),
  store_region VARCHAR(50)
);

CREATE TABLE dim_dates (
  date_id DATE PRIMARY KEY,
  year INT,
  month INT,
  day INT,
  day_of_week INT,
  quarter INT,
  week_of_year INT,
  is_weekend BOOLEAN,
  is_holiday BOOLEAN
);

-- ==========================================
-- Exploratory Data Analysis Queries
-- ==========================================

-- Basic summary statistics
-- Run this query to get an overview of your sales data
SELECT 
  COUNT(*) AS total_transactions,
  COUNT(DISTINCT customer_name) AS unique_customers,
  COUNT(DISTINCT product_name) AS unique_products,
  SUM(quantity_sold) AS total_items_sold,
  SUM(total_amount) AS total_revenue,
  AVG(total_amount) AS avg_transaction_value,
  MIN(date_of_sale) AS earliest_date,
  MAX(date_of_sale) AS latest_date
FROM sales_transactions;

-- Summary statistics for numeric columns
-- This query provides detailed statistics for key numeric metrics
SELECT
  'Quantity Sold' AS metric,
  AVG(quantity_sold) AS mean,
  (SELECT quantity_sold FROM sales_transactions ORDER BY quantity_sold LIMIT 1 OFFSET (SELECT COUNT(*)/2 FROM sales_transactions)) AS median,
  STDDEV(quantity_sold) AS std_dev,
  MIN(quantity_sold) AS min_value,
  MAX(quantity_sold) AS max_value
FROM sales_transactions
UNION ALL
SELECT
  'Price' AS metric,
  AVG(price) AS mean,
  (SELECT price FROM sales_transactions ORDER BY price LIMIT 1 OFFSET (SELECT COUNT(*)/2 FROM sales_transactions)) AS median,
  STDDEV(price) AS std_dev,
  MIN(price) AS min_value,
  MAX(price) AS max_value
FROM sales_transactions
UNION ALL
SELECT
  'Total Amount' AS metric,
  AVG(total_amount) AS mean,
  (SELECT total_amount FROM sales_transactions ORDER BY total_amount LIMIT 1 OFFSET (SELECT COUNT(*)/2 FROM sales_transactions)) AS median,
  STDDEV(total_amount) AS std_dev,
  MIN(total_amount) AS min_value,
  MAX(total_amount) AS max_value
FROM sales_transactions;

-- Missing values analysis
-- Identify and quantify missing data in your dataset
SELECT
  SUM(CASE WHEN quantity_sold IS NULL THEN 1 ELSE 0 END) AS quantity_sold_nulls,
  SUM(CASE WHEN price IS NULL THEN 1 ELSE 0 END) AS price_nulls,
  SUM(CASE WHEN total_amount IS NULL THEN 1 ELSE 0 END) AS total_amount_nulls,
  COUNT(*) AS total_rows,
  (SUM(CASE WHEN quantity_sold IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) AS quantity_sold_null_pct,
  (SUM(CASE WHEN price IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) AS price_null_pct,
  (SUM(CASE WHEN total_amount IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) AS total_amount_null_pct
FROM sales_transactions;

-- ==========================================
-- Time Series Analysis Queries
-- ==========================================

-- Daily sales aggregation
-- Track sales performance by day
SELECT 
  DATE(date_of_sale) AS sale_date,
  SUM(total_amount) AS daily_sales
FROM sales_transactions
GROUP BY DATE(date_of_sale)
ORDER BY sale_date;

-- Monthly sales aggregation
-- Track sales performance by month
SELECT 
  YEAR(date_of_sale) AS year,
  MONTH(date_of_sale) AS month,
  SUM(total_amount) AS monthly_sales
FROM sales_transactions
GROUP BY YEAR(date_of_sale), MONTH(date_of_sale)
ORDER BY year, month;

-- Year-over-year comparison by month
-- Compare monthly performance across years
WITH monthly_sales AS (
  SELECT 
    YEAR(date_of_sale) AS year,
    MONTH(date_of_sale) AS month,
    SUM(total_amount) AS monthly_sales
  FROM sales_transactions
  GROUP BY YEAR(date_of_sale), MONTH(date_of_sale)
)
SELECT 
  m1.year,
  m1.month,
  m1.monthly_sales,
  m2.monthly_sales AS prev_year_sales,
  (m1.monthly_sales - m2.monthly_sales) AS yoy_difference,
  CASE 
    WHEN m2.monthly_sales = 0 THEN NULL
    ELSE ((m1.monthly_sales - m2.monthly_sales) / m2.monthly_sales) * 100 
  END AS yoy_growth_pct
FROM monthly_sales m1
LEFT JOIN monthly_sales m2 ON m1.year = m2.year + 1 AND m1.month = m2.month
ORDER BY m1.year, m1.month;

-- Moving averages for trend analysis
-- Calculate 7-day and 30-day moving averages to identify trends
SELECT 
  DATE(date_of_sale) AS sale_date,
  SUM(total_amount) AS daily_sales,
  (
    SELECT AVG(daily_total)
    FROM (
      SELECT 
        DATE(date_of_sale) AS inner_date,
        SUM(total_amount) AS daily_total
      FROM sales_transactions
      WHERE DATE(date_of_sale) BETWEEN DATE_SUB(DATE(s.date_of_sale), INTERVAL 6 DAY) AND DATE(s.date_of_sale)
      GROUP BY DATE(date_of_sale)
    ) AS t
  ) AS seven_day_moving_avg,
  (
    SELECT AVG(daily_total)
    FROM (
      SELECT 
        DATE(date_of_sale) AS inner_date,
        SUM(total_amount) AS daily_total
      FROM sales_transactions
      WHERE DATE(date_of_sale) BETWEEN DATE_SUB(DATE(s.date_of_sale), INTERVAL 29 DAY) AND DATE(s.date_of_sale)
      GROUP BY DATE(date_of_sale)
    ) AS t
  ) AS thirty_day_moving_avg
FROM sales_transactions s
GROUP BY DATE(date_of_sale)
ORDER BY sale_date;

-- Seasonality analysis by day of week
-- Identify which days of the week perform best
SELECT 
  DAYOFWEEK(date_of_sale) AS day_of_week,
  CASE DAYOFWEEK(date_of_sale)
    WHEN 1 THEN 'Sunday'
    WHEN 2 THEN 'Monday'
    WHEN 3 THEN 'Tuesday'
    WHEN 4 THEN 'Wednesday'
    WHEN 5 THEN 'Thursday'
    WHEN 6 THEN 'Friday'
    WHEN 7 THEN 'Saturday'
  END AS day_name,
  AVG(daily_total) AS avg_daily_sales
FROM (
  SELECT 
    DATE(date_of_sale) AS sale_date,
    DAYOFWEEK(date_of_sale) AS day_of_week,
    SUM(total_amount) AS daily_total
  FROM sales_transactions
  GROUP BY DATE(date_of_sale), DAYOFWEEK(date_of_sale)
) AS daily_sales
GROUP BY day_of_week, day_name
ORDER BY day_of_week;

-- Seasonality analysis by month
-- Identify which months perform best
SELECT 
  MONTH(date_of_sale) AS month,
  CASE MONTH(date_of_sale)
    WHEN 1 THEN 'January'
    WHEN 2 THEN 'February'
    WHEN 3 THEN 'March'
    WHEN 4 THEN 'April'
    WHEN 5 THEN 'May'
    WHEN 6 THEN 'June'
    WHEN 7 THEN 'July'
    WHEN 8 THEN 'August'
    WHEN 9 THEN 'September'
    WHEN 10 THEN 'October'
    WHEN 11 THEN 'November'
    WHEN 12 THEN 'December'
  END AS month_name,
  AVG(monthly_total) AS avg_monthly_sales
FROM (
  SELECT 
    YEAR(date_of_sale) AS year,
    MONTH(date_of_sale) AS month,
    SUM(total_amount) AS monthly_total
  FROM sales_transactions
  GROUP BY YEAR(date_of_sale), MONTH(date_of_sale)
) AS monthly_sales
GROUP BY month, month_name
ORDER BY month;

-- ==========================================
-- Customer Segmentation Queries
-- ==========================================

-- Customer purchase metrics for segmentation
-- Calculate key metrics for each customer to enable segmentation
WITH customer_metrics AS (
  SELECT 
    customer_name,
    COUNT(*) AS transaction_count,
    SUM(total_amount) AS total_spend,
    AVG(total_amount) AS avg_transaction_value,
    SUM(quantity_sold) AS total_items,
    AVG(quantity_sold) AS avg_items_per_transaction,
    SUM(total_amount) / SUM(quantity_sold) AS avg_spend_per_item,
    MAX(date_of_sale) AS last_purchase_date,
    DATEDIFF(CURRENT_DATE(), MAX(date_of_sale)) AS days_since_last_purchase
  FROM sales_transactions
  GROUP BY customer_name
)
SELECT 
  customer_name,
  transaction_count,
  total_spend,
  avg_transaction_value,
  total_items,
  avg_items_per_transaction,
  avg_spend_per_item,
  last_purchase_date,
  days_since_last_purchase,
  -- RFM (Recency, Frequency, Monetary) segmentation
  CASE 
    WHEN days_since_last_purchase <= 30 AND transaction_count >= 10 AND total_spend >= 1000 THEN 'High-Value Loyal'
    WHEN days_since_last_purchase <= 30 AND transaction_count >= 10 THEN 'Loyal'
    WHEN days_since_last_purchase <= 30 AND total_spend >= 1000 THEN 'Big Spender'
    WHEN days_since_last_purchase <= 30 THEN 'Recent Customer'
    WHEN transaction_count >= 10 THEN 'Frequent Customer'
    WHEN total_spend >= 1000 THEN 'High Spender'
    WHEN days_since_last_purchase > 180 THEN 'Churned'
    ELSE 'Average Customer'
  END AS customer_segment
FROM customer_metrics
ORDER BY total_spend DESC;

-- Customer segment summary
-- Summarize the characteristics of each customer segment
WITH customer_segments AS (
  SELECT 
    customer_name,
    COUNT(*) AS transaction_count,
    SUM(total_amount) AS total_spend,
    AVG(total_amount) AS avg_transaction_value,
    MAX(date_of_sale) AS last_purchase_date,
    DATEDIFF(CURRENT_DATE(), MAX(date_of_sale)) AS days_since_last_purchase,
    CASE 
      WHEN DATEDIFF(CURRENT_DATE(), MAX(date_of_sale)) <= 30 AND COUNT(*) >= 10 AND SUM(total_amount) >= 1000 THEN 'High-Value Loyal'
      WHEN DATEDIFF(CURRENT_DATE(), MAX(date_of_sale)) <= 30 AND COUNT(*) >= 10 THEN 'Loyal'
      WHEN DATEDIFF(CURRENT_DATE(), MAX(date_of_sale)) <= 30 AND SUM(total_amount) >= 1000 THEN 'Big Spender'
      WHEN DATEDIFF(CURRENT_DATE(), MAX(date_of_sale)) <= 30 THEN 'Recent Customer'
      WHEN COUNT(*) >= 10 THEN 'Frequent Customer'
      WHEN SUM(total_amount) >= 1000 THEN 'High Spender'
      WHEN DATEDIFF(CURRENT_DATE(), MAX(date_of_sale)) > 180 THEN 'Churned'
      ELSE 'Average Customer'
    END AS customer_segment
  FROM sales_transactions
  GROUP BY customer_name
)
SELECT 
  customer_segment,
  COUNT(*) AS segment_size,
  AVG(total_spend) AS avg_total_spend,
  AVG(transaction_count) AS avg_transaction_count,
  AVG(avg_transaction_value) AS avg_transaction_value,
  AVG(days_since_last_purchase) AS avg_days_since_last_purchase
FROM customer_segments
GROUP BY customer_segment
ORDER BY AVG(total_spend) DESC;

-- ==========================================
-- Anomaly Detection Queries
-- ==========================================

-- Identify daily sales anomalies using statistical methods (IQR method)
-- Detect outliers in daily sales using the Interquartile Range method
WITH daily_sales AS (
  SELECT 
    DATE(date_of_sale) AS sale_date,
    SUM(total_amount) AS daily_total
  FROM sales_transactions
  GROUP BY DATE(date_of_sale)
),
quartiles AS (
  SELECT 
    @row_num:=@row_num+1 as row_num,
    daily_total
  FROM daily_sales, (SELECT @row_num:=0) r
  ORDER BY daily_total
),
stats AS (
  SELECT
    (SELECT daily_total FROM quartiles WHERE row_num = FLOOR((SELECT COUNT(*) FROM quartiles) * 0.25)) AS q1,
    (SELECT daily_total FROM quartiles WHERE row_num = FLOOR((SELECT COUNT(*) FROM quartiles) * 0.75)) AS q3
)
SELECT 
  ds.sale_date,
  ds.daily_total,
  CASE 
    WHEN ds.daily_total > (s.q3 + 1.5 * (s.q3 - s.q1)) THEN 'High Outlier'
    WHEN ds.daily_total < (s.q1 - 1.5 * (s.q3 - s.q1)) THEN 'Low Outlier'
    ELSE 'Normal'
  END AS anomaly_status,
  s.q1 AS first_quartile,
  s.q3 AS third_quartile,
  (s.q3 - s.q1) AS interquartile_range,
  (s.q1 - 1.5 * (s.q3 - s.q1)) AS lower_bound,
  (s.q3 + 1.5 * (s.q3 - s.q1)) AS upper_bound
FROM daily_sales ds, stats s
WHERE ds.daily_total > (s.q3 + 1.5 * (s.q3 - s.q1)) OR ds.daily_total < (s.q1 - 1.5 * (s.q3 - s.q1))
ORDER BY ds.sale_date;

-- Z-score method for anomaly detection
-- Detect outliers in daily sales using the Z-score method
WITH daily_sales AS (
  SELECT 
    DATE(date_of_sale) AS sale_date,
    SUM(total_amount) AS daily_total
  FROM sales_transactions
  GROUP BY DATE(date_of_sale)
),
stats AS (
  SELECT 
    AVG(daily_total) AS mean,
    STDDEV(daily_total) AS std_dev
  FROM daily_sales
)
SELECT 
  ds.sale_date,
  ds.daily_total,
  (ds.daily_total - s.mean) / s.std_dev AS z_score,
  CASE 
    WHEN ABS((ds.daily_total - s.mean) / s.std_dev) > 2.5 THEN 'Anomaly'
    ELSE 'Normal'
  END AS anomaly_status
FROM daily_sales ds, stats s
WHERE ABS((ds.daily_total - s.mean) / s.std_dev) > 2.5
ORDER BY ABS((ds.daily_total - s.mean) / s.std_dev) DESC;

-- ==========================================
-- Feature Importance Analysis Queries
-- ==========================================

-- Product category impact on sales
-- Analyze how different product categories perform
SELECT 
  product_category,
  COUNT(*) AS transaction_count,
  SUM(total_amount) AS total_sales,
  AVG(total_amount) AS avg_transaction_value,
  SUM(quantity_sold) AS total_quantity,
  AVG(quantity_sold) AS avg_quantity,
  SUM(total_amount) / SUM(quantity_sold) AS avg_price
FROM sales_transactions
GROUP BY product_category
ORDER BY total_sales DESC;

-- Day of week impact on sales
-- Analyze sales performance by day of week
SELECT 
  DAYOFWEEK(date_of_sale) AS day_of_week,
  CASE DAYOFWEEK(date_of_sale)
    WHEN 1 THEN 'Sunday'
    WHEN 2 THEN 'Monday'
    WHEN 3 THEN 'Tuesday'
    WHEN 4 THEN 'Wednesday'
    WHEN 5 THEN 'Thursday'
    WHEN 6 THEN 'Friday'
    WHEN 7 THEN 'Saturday'
  END AS day_name,
  COUNT(*) AS transaction_count,
  SUM(total_amount) AS total_sales,
  AVG(total_amount) AS avg_transaction_value
FROM sales_transactions
GROUP BY day_of_week, day_name
ORDER BY total_sales DESC;

-- Month impact on sales
-- Analyze sales performance by month
SELECT 
  MONTH(date_of_sale) AS month,
  CASE MONTH(date_of_sale)
    WHEN 1 THEN 'January'
    WHEN 2 THEN 'February'
    WHEN 3 THEN 'March'
    WHEN 4 THEN 'April'
    WHEN 5 THEN 'May'
    WHEN 6 THEN 'June'
    WHEN 7 THEN 'July'
    WHEN 8 THEN 'August'
    WHEN 9 THEN 'September'
    WHEN 10 THEN 'October'
    WHEN 11 THEN 'November'
    WHEN 12 THEN 'December'
  END AS month_name,
  COUNT(*) AS transaction_count,
  SUM(total_amount) AS total_sales,
  AVG(total_amount) AS avg_transaction_value
FROM sales_transactions
GROUP BY month, month_name
ORDER BY month;

-- Store performance comparison
-- Compare performance across different stores
SELECT 
  store_id,
  COUNT(*) AS transaction_count,
  SUM(total_amount) AS total_sales,
  AVG(total_amount) AS avg_transaction_value,
  SUM(quantity_sold) AS total_quantity
FROM sales_transactions
GROUP BY store_id
ORDER BY total_sales DESC;

-- Payment method analysis
-- Analyze sales by payment method
SELECT 
  payment_method,
  COUNT(*) AS transaction_count,
  SUM(total_amount) AS total_sales,
  AVG(total_amount) AS avg_transaction_value
FROM sales_transactions
GROUP BY payment_method
ORDER BY total_sales DESC;

-- Price point analysis
-- Analyze sales performance by price range
WITH price_buckets AS (
  SELECT 
    CASE 
      WHEN price < 10 THEN 'Under $10'
      WHEN price >= 10 AND price < 25 THEN '$10-$25'
      WHEN price >= 25 AND price < 50 THEN '$25-$50'
      WHEN price >= 50 AND price < 100 THEN '$50-$100'
      ELSE 'Over $100'
    END AS price_range,
    quantity_sold,
    total_amount
  FROM sales_transactions
)
SELECT 
  price_range,
  COUNT(*) AS transaction_count,
  SUM(quantity_sold) AS total_quantity,
  SUM(total_amount) AS total_sales,
  AVG(total_amount) AS avg_transaction_value
FROM price_buckets
GROUP BY price_range
ORDER BY CASE 
  WHEN price_range = 'Under $10' THEN 1
  WHEN price_range = '$10-$25' THEN 2
  WHEN price_range = '$25-$50' THEN 3
  WHEN price_range = '$50-$100' THEN 4
  ELSE 5
END;

-- ==========================================
-- Sales Forecasting Preparation Queries
-- ==========================================

-- Create a complete date dimension table for forecasting
-- This procedure populates the date dimension table
CREATE PROCEDURE sp_populate_date_dimension(start_date DATE, end_date DATE)
BEGIN
  DECLARE curr_date DATE;
  SET curr_date = start_date;
  
  -- Clear existing data
  TRUNCATE TABLE dim_dates;
  
  -- Loop through dates and populate the table
  WHILE curr_date <= end_date DO
    INSERT INTO dim_dates (
      date_id,
      year,
      month,
      day,
      day_of_week,
      quarter,
      week_of_year,
      is_weekend
    )
    VALUES (
      curr_date,
      YEAR(curr_date),
      MONTH(curr_date),
      DAY(curr_date),
      DAYOFWEEK(curr_date),
      QUARTER(curr_date),
      WEEKOFYEAR(curr_date),
      CASE WHEN DAYOFWEEK(curr_date) IN (1, 7) THEN TRUE ELSE FALSE END
    );
    
    SET curr_date = DATE_ADD(curr_date, INTERVAL 1 DAY);
  END WHILE;
END //

-- Create a complete sales time series with zero-filling for missing dates
-- This query ensures a complete time series with no gaps
SELECT 
  dd.date_id AS sale_date,
  COALESCE(ds.daily_total, 0) AS daily_sales
FROM dim_dates dd
LEFT JOIN (
  SELECT 
    DATE(date_of_sale) AS sale_date,
    SUM(total_amount) AS daily_total
  FROM sales_transactions
  GROUP BY DATE(date_of_sale)
) ds ON dd.date_id = ds.sale_date
WHERE dd.date_id BETWEEN 
  (SELECT MIN(DATE(date_of_sale)) FROM sales_transactions) AND
  (SELECT MAX(DATE(date_of_sale)) FROM sales_transactions)
ORDER BY dd.date_id;

-- Create lagged features for time series forecasting
-- Generate lag features for machine learning models
WITH daily_sales AS (
  SELECT 
    DATE(date_of_sale) AS sale_date,
    SUM(total_amount) AS daily_total
  FROM sales_transactions
  GROUP BY DATE(date_of_sale)
  ORDER BY sale_date
)
SELECT 
  d1.sale_date,
  d1.daily_total,
  d2.daily_total AS sales_lag1,
  d3.daily_total AS sales_lag2,
  d4.daily_total AS sales_lag3,
  d8.daily_total AS sales_lag7,
  d15.daily_total AS sales_lag14,
  d31.daily_total AS sales_lag30,
  (
    SELECT AVG(daily_total)
    FROM daily_sales d_inner
    WHERE d_inner.sale_date BETWEEN DATE_SUB(d1.sale_date, INTERVAL 7 DAY) AND DATE_SUB(d1.sale_date, INTERVAL 1 DAY)
  ) AS moving_avg_7day,
  (
    SELECT AVG(daily_total)
    FROM daily_sales d_inner
    WHERE d_inner.sale_date BETWEEN DATE_SUB(d1.sale_date, INTERVAL 30 DAY) AND DATE_SUB(d1.sale_date, INTERVAL 1 DAY)
  ) AS moving_avg_30day
FROM daily_sales d1
LEFT JOIN daily_sales d2 ON d1.sale_date = DATE_ADD(d2.sale_date, INTERVAL 1 DAY)
LEFT JOIN daily_sales d3 ON d1.sale_date = DATE_ADD(d3.sale_date, INTERVAL 2 DAY)
LEFT JOIN daily_sales d4 ON d1.sale_date = DATE_ADD(d4.sale_date, INTERVAL 3 DAY)
LEFT JOIN daily_sales d8 ON d1.sale_date = DATE_ADD(d8.sale_date, INTERVAL 7 DAY)
LEFT JOIN daily_sales d15 ON d1.sale_date = DATE_ADD(d15.sale_date, INTERVAL 14 DAY)
LEFT JOIN daily_sales d31 ON d1.sale_date = DATE_ADD(d31.sale_date, INTERVAL 30 DAY)
ORDER BY d1.sale_date;

-- ==========================================
-- Dashboard Views
-- ==========================================

-- 1. Daily Sales Overview
CREATE VIEW vw_daily_sales_overview AS
SELECT 
  DATE(date_of_sale) AS sale_date,
  COUNT(*) AS transaction_count,
  SUM(quantity_sold) AS total_items_sold,
  SUM(total_amount) AS total_sales,
  AVG(total_amount) AS avg_transaction_value,
  COUNT(DISTINCT customer_name) AS unique_customers
FROM sales_transactions
GROUP BY DATE(date_of_sale);

-- 2. Monthly Sales Trends
CREATE VIEW vw_monthly_sales_trends AS
SELECT 
  YEAR(date_of_sale) AS year,
  MONTH(date_of_sale) AS month,
  COUNT(*) AS transaction_count,
  SUM(quantity_sold) AS total_items_sold,
  SUM(total_amount) AS total_sales,
  AVG(total_amount) AS avg_transaction_value,
  COUNT(DISTINCT customer_name) AS unique_customers
FROM sales_transactions
GROUP BY YEAR(date_of_sale), MONTH(date_of_sale);

-- 3. Product Category Performance
CREATE VIEW vw_product_category_performance AS
SELECT 
  product_category,
  COUNT(*) AS transaction_count,
  SUM(quantity_sold) AS total_items_sold,
  SUM(total_amount) AS total_sales,
  AVG(price) AS avg_price,
  AVG(total_amount) AS avg_transaction_value,
  COUNT(DISTINCT customer_name) AS unique_customers
FROM sales_transactions
GROUP BY product_category;

-- 4. Customer Purchase Patterns
CREATE VIEW vw_customer_purchase_patterns AS
SELECT 
  customer_name,
  COUNT(*) AS transaction_count,
  SUM(quantity_sold) AS total_items_purchased,
  SUM(total_amount) AS total_spend,
  AVG(total_amount) AS avg_transaction_value,
  MIN(date_of_sale) AS first_purchase_date,
  MAX(date_of_sale) AS last_purchase_date,
  DATEDIFF(MAX(date_of_sale), MIN(date_of_sale)) AS customer_lifespan_days
FROM sales_transactions
GROUP BY customer_name;

-- 5. Store Performance Comparison
CREATE VIEW vw_store_performance AS
SELECT 
  store_id,
  COUNT(*) AS transaction_count,
  SUM(quantity_sold) AS total_items_sold,
  SUM(total_amount) AS total_sales,
  AVG(total_amount) AS avg_transaction_value,
  COUNT(DISTINCT customer_name) AS unique_customers,
  SUM(total_amount) / COUNT(DISTINCT customer_name) AS revenue_per_customer
FROM sales_transactions
GROUP BY store_id;

-- 6. Sales by Day of Week
CREATE VIEW vw_sales_by_day_of_week AS
SELECT 
  DAYOFWEEK(date_of_sale) AS day_of_week,
  CASE DAYOFWEEK(date_of_sale)
    WHEN 1 THEN 'Sunday'
    WHEN 2 THEN 'Monday'
    WHEN 3 THEN 'Tuesday'
    WHEN 4 THEN 'Wednesday'
    WHEN 5 THEN 'Thursday'
    WHEN 6 THEN 'Friday'
    WHEN 7 THEN 'Saturday'
  END AS day_name,
  COUNT(*) AS transaction_count,
  SUM(quantity_sold) AS total_items_sold,
  SUM(total_amount) AS total_sales,
  AVG(total_amount) AS avg_transaction_value
FROM sales_transactions
GROUP BY day_of_week, day_name;

-- 7. Sales by Hour of Day
CREATE VIEW vw_sales_by_hour AS
SELECT 
  HOUR(date_of_sale) AS hour_of_day,
  COUNT(*) AS transaction_count,
  SUM(quantity_sold) AS total_items_sold,
  SUM(total_amount) AS total_sales,
  AVG(total_amount) AS avg_transaction_value
FROM sales_transactions
GROUP BY hour_of_day;

-- 8. Top Selling Products
CREATE VIEW vw_top_selling_products AS
SELECT 
  product_name,
  product_category,
  COUNT(*) AS transaction_count,
  SUM(quantity_sold) AS total_quantity_sold,
  SUM(total_amount) AS total_sales,
  AVG(price) AS avg_price
FROM sales_transactions
GROUP BY product_name, product_category
ORDER BY total_quantity_sold DESC;

-- 9. Top Customers
CREATE VIEW vw_top_customers AS
SELECT 
  customer_name,
  COUNT(*) AS transaction_count,
  SUM(quantity_sold) AS total_items_purchased,
  SUM(total_amount) AS total_spend,
  AVG(total_amount) AS avg_transaction_value,
  MAX(date_of_sale) AS last_purchase_date
FROM sales_transactions
GROUP BY customer_name
ORDER BY total_spend DESC;

-- 10. Sales Anomalies
CREATE VIEW vw_sales_anomalies AS
WITH daily_sales AS (
  SELECT 
    DATE(date_of_sale) AS sale_date,
    SUM(total_amount) AS daily_total
  FROM sales_transactions
  GROUP BY DATE(date_of_sale)
),
stats AS (
  SELECT 
    AVG(daily_total) AS mean,
    STDDEV(daily_total) AS std_dev
  FROM daily_sales
)
SELECT 
  ds.sale_date,
  ds.daily_total,
  (ds.daily_total - s.mean) / s.std_dev AS z_score,
  CASE 
    WHEN ABS((ds.daily_total - s.mean) / s.std_dev) > 2.5 THEN 'Anomaly'
    ELSE 'Normal'
  END AS anomaly_status
FROM daily_sales ds, stats s
WHERE ABS((ds.daily_total - s.mean) / s.std_dev) > 2.5
ORDER BY ABS((ds.daily_total - s.mean) / s.std_dev) DESC;

-- ==========================================
-- Stored Procedures for Analytics Tasks
-- ==========================================

-- 1. Get sales forecast for next N days
CREATE PROCEDURE sp_get_sales_forecast(IN days_ahead INT)
BEGIN
  -- This is a simplified forecast using moving averages
  -- In a real implementation, you would use more sophisticated methods
  WITH daily_sales AS (
    SELECT 
      DATE(date_of_sale) AS sale_date,
      SUM(total_amount) AS daily_total
    FROM sales_transactions
    GROUP BY DATE(date_of_sale)
    ORDER BY sale_date
  ),
  last_30_days AS (
    SELECT AVG(daily_total) AS avg_daily_sales
    FROM (
      SELECT daily_total
      FROM daily_sales
      ORDER BY sale_date DESC
      LIMIT 30
    ) AS recent
  ),
  last_7_days AS (
    SELECT AVG(daily_total) AS avg_daily_sales
    FROM (
      SELECT daily_total
      FROM daily_sales
      ORDER BY sale_date DESC
      LIMIT 7
    ) AS recent
  ),
  date_series AS (
    SELECT 
      DATE_ADD((SELECT MAX(sale_date) FROM daily_sales), INTERVAL seq DAY) AS forecast_date
    FROM 
      (SELECT 0 AS seq UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION
       SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION
       SELECT 10 UNION SELECT 11 UNION SELECT 12 UNION SELECT 13 UNION SELECT 14 UNION
       SELECT 15 UNION SELECT 16 UNION SELECT 17 UNION SELECT 18 UNION SELECT 19 UNION
       SELECT 20 UNION SELECT 21 UNION SELECT 22 UNION SELECT 23 UNION SELECT 24 UNION
       SELECT 25 UNION SELECT 26 UNION SELECT 27 UNION SELECT 28 UNION SELECT 29) AS sequence
    WHERE seq < days_ahead
  )
  SELECT 
    ds.forecast_date,
    l7.avg_daily_sales AS forecast_7day_avg,
    l30.avg_daily_sales AS forecast_30day_avg,
    (l7.avg_daily_sales * 0.7 + l30.avg_daily_sales * 0.3) AS weighted_forecast
  FROM date_series ds, last_7_days l7, last_30_days l30;
END //

-- 2. Get customer lifetime value
CREATE PROCEDURE sp_get_customer_ltv()
BEGIN
  WITH customer_purchases AS (
    SELECT 
      customer_name,
      COUNT(*) AS transaction_count,
      SUM(total_amount) AS total_spend,
      MIN(date_of_sale) AS first_purchase_date,
      MAX(date_of_sale) AS last_purchase_date,
      DATEDIFF(MAX(date_of_sale), MIN(date_of_sale)) AS customer_lifespan_days
    FROM sales_transactions
    GROUP BY customer_name
  )
  SELECT 
    customer_name,
    transaction_count,
    total_spend,
    customer_lifespan_days,
    CASE 
      WHEN customer_lifespan_days = 0 THEN total_spend
      ELSE total_spend / customer_lifespan_days
    END AS spend_per_day,
    CASE 
      WHEN customer_lifespan_days = 0 THEN total_spend * 365
      ELSE (total_spend / customer_lifespan_days) * 365
    END AS annual_value,
    CASE 
      WHEN customer_lifespan_days = 0 THEN total_spend * 365 * 3
      ELSE ((total_spend / customer_lifespan_days) * 365) * 3
    END AS estimated_3year_ltv -- Using a simple 3-year projection
  FROM customer_purchases
  ORDER BY estimated_3year_ltv DESC;
END //

-- 3. Identify products frequently purchased together
CREATE PROCEDURE sp_product_affinity_analysis()
BEGIN
  WITH product_pairs AS (
    SELECT 
      a.transaction_id,
      a.product_name AS product_a,
      b.product_name AS product_b
    FROM sales_transactions a
    JOIN sales_transactions b ON a.transaction_id = b.transaction_id AND a.product_name < b.product_name
  )
  SELECT 
    product_a,
    product_b,
    COUNT(*) AS pair_frequency,
    COUNT(*) * 100.0 / (
      SELECT COUNT(DISTINCT transaction_id) FROM sales_transactions
    ) AS pct_of_transactions
  FROM product_pairs
  GROUP BY product_a, product_b
  HAVING COUNT(*) > 5
  ORDER BY pair_frequency DESC;
END //

-- 4. Generate RFM (Recency, Frequency, Monetary) customer segments
CREATE PROCEDURE sp_generate_rfm_segments()
BEGIN
  WITH customer_rfm AS (
    SELECT 
      customer_name,
      DATEDIFF(CURRENT_DATE(), MAX(date_of_sale)) AS recency,
      COUNT(*) AS frequency,
      SUM(total_amount) AS monetary
    FROM sales_transactions
    GROUP BY customer_name
  ),
  rfm_scores AS (
    SELECT 
      customer_name,
      recency,
      frequency,
      monetary,
      NTILE(5) OVER (ORDER BY recency DESC) AS r_score,
      NTILE(5) OVER (ORDER BY frequency) AS f_score,
      NTILE(5) OVER (ORDER BY monetary) AS m_score
    FROM customer_rfm
  )
  SELECT 
    customer_name,
    recency,
    frequency,
    monetary,
    r_score,
    f_score,
    m_score,
    CONCAT(r_score, f_score, m_score) AS rfm_score,
    CASE 
      WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
      WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'Loyal Customers'
      WHEN r_score >= 3 AND f_score >= 1 AND m_score >= 2 THEN 'Potential Loyalists'
      WHEN r_score >= 4 AND f_score <= 2 AND m_score <= 2 THEN 'New Customers'
      WHEN r_score <= 2 AND f_score >= 3 AND m_score >= 3 THEN 'At Risk Customers'
      WHEN r_score <= 2 AND f_score >= 2 AND m_score >= 2 THEN 'Need Attention'
      WHEN r_score <= 1 AND f_score >= 4 AND m_score >= 4 THEN 'Cannot Lose Them'
      WHEN r_score <= 2 AND f_score <= 2 AND m_score <= 2 THEN 'Hibernating'
      WHEN r_score <= 1 AND f_score <= 1 AND m_score <= 1 THEN 'Lost'
      ELSE 'Others'
    END AS customer_segment
  FROM rfm_scores
  ORDER BY 
    CASE 
      WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 1
      WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 2
      ELSE 3
    END;
END //

-- 5. Detect sales anomalies
CREATE PROCEDURE sp_detect_sales_anomalies(IN sensitivity FLOAT)
BEGIN
  -- sensitivity parameter controls threshold (default 2.5 standard deviations)
  WITH daily_sales AS (
    SELECT 
      DATE(date_of_sale) AS sale_date,
      SUM(total_amount) AS daily_total
    FROM sales_transactions
    GROUP BY DATE(date_of_sale)
  ),
  stats AS (
    SELECT 
      AVG(daily_total) AS mean,
      STDDEV(daily_total) AS std_dev
    FROM daily_sales
  )
  SELECT 
    ds.sale_date,
    ds.daily_total,
    (ds.daily_total - s.mean) / s.std_dev AS z_score,
    CASE 
      WHEN ABS((ds.daily_total - s.mean) / s.std_dev) > sensitivity THEN 'Anomaly'
      ELSE 'Normal'
    END AS anomaly_status
  FROM daily_sales ds, stats s
  WHERE ABS((ds.daily_total - s.mean) / s.std_dev) > sensitivity
  ORDER BY ABS((ds.daily_total - s.mean) / s.std_dev) DESC;
END //

-- Reset delimiter
DELIMITER ;

-- ==========================================
-- Sample Data Insertion (for testing)
-- ==========================================

-- Uncomment and modify this section to insert sample data for testing
/*
-- Insert sample data into sales_transactions
INSERT INTO sales_transactions 
(transaction_id, customer_name, employee_name, date_of_sale, product_id, product_name, 
 product_category, quantity_sold, price, total_amount, store_id, payment_method)
VALUES
('T001', 'John Smith', 'Mary Johnson', '2023-01-01 10:30:00', 'P001', 'T-Shirt', 'Clothing', 2, 19.99, 39.98, 'S001', 'Credit Card'),
('T002', 'Jane Doe', 'Robert Brown', '2023-01-01 11:45:00', 'P002', 'Jeans', 'Clothing', 1, 49.99, 49.99, 'S001', 'Cash'),
('T003', 'Michael Wilson', 'Mary Johnson', '2023-01-02 09:15:00', 'P003', 'Sneakers', 'Footwear', 1, 79.99, 79.99, 'S002', 'Credit Card'),
('T004', 'Sarah Johnson', 'David Miller', '2023-01-02 14:20:00', 'P001', 'T-Shirt', 'Clothing', 3, 19.99, 59.97, 'S002', 'Debit Card'),
('T005', 'Robert Brown', 'Jennifer Davis', '2023-01-03 16:30:00', 'P004', 'Backpack', 'Accessories', 1, 39.99, 39.99, 'S001', 'Credit Card');

-- Populate date dimension
CALL sp_populate_date_dimension('2023-01-01', '2023-12-31');
*/

-- ==========================================
-- End of Script
-- ==========================================