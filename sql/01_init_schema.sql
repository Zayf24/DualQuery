-- ==============================================================================
-- DualQuery Core Schema (DuckDB)
-- Supports Relational OLAP, HNSW Vector Similarity (VSS), and BM25 Full-Text Search (FTS)
-- ==============================================================================

-- 1. Load Required Extensions
INSTALL vss;
LOAD vss;
INSTALL fts;
LOAD fts;

-- Enable experimental persistence for HNSW indexes on disk
SET hnsw_enable_experimental_persistence = true;

-- 2. Core Relational Tables (Olist E-Commerce Schema)

CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR PRIMARY KEY,
    customer_unique_id VARCHAR NOT NULL,
    customer_zip_code_prefix VARCHAR,
    customer_city VARCHAR,
    customer_state VARCHAR(2)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR PRIMARY KEY,
    customer_id VARCHAR REFERENCES customers(customer_id),
    order_status VARCHAR,
    order_purchase_timestamp TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP,
    delivery_delay_days INTEGER GENERATED ALWAYS AS (
        date_diff('day', order_estimated_delivery_date, order_delivered_customer_date)
    ) VIRTUAL
);

CREATE TABLE IF NOT EXISTS order_payments (
    order_id VARCHAR REFERENCES orders(order_id),
    payment_sequential INTEGER,
    payment_type VARCHAR,
    payment_installments INTEGER,
    payment_value DECIMAL(10, 2),
    PRIMARY KEY (order_id, payment_sequential)
);

CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR PRIMARY KEY,
    product_category_name VARCHAR,
    product_category_name_english VARCHAR,
    product_weight_g INTEGER
);

CREATE TABLE IF NOT EXISTS order_items (
    order_id VARCHAR REFERENCES orders(order_id),
    order_item_id INTEGER,
    product_id VARCHAR REFERENCES products(product_id),
    seller_id VARCHAR,
    price DECIMAL(10, 2),
    freight_value DECIMAL(10, 2),
    PRIMARY KEY (order_id, order_item_id)
);

-- 3. Customer Reviews with Unified 768-d Embeddings (Compatible with EmbeddingGemma 2 & BGE-Base)

CREATE TABLE IF NOT EXISTS order_reviews (
    review_id VARCHAR PRIMARY KEY,
    order_id VARCHAR REFERENCES orders(order_id),
    review_score TINYINT,
    review_comment_title VARCHAR,
    review_comment_message VARCHAR,
    review_creation_date TIMESTAMP,
    embedding FLOAT[768]
);

-- 4. Native HNSW Vector Index (Plan B Pushdown)
-- metric = 'cosine': optimizes normalized dot-product / angular distance
-- ef_construction = 128: search depth during index construction for high recall
-- M = 16: number of bidirectional connections per graph layer
CREATE INDEX IF NOT EXISTS idx_reviews_embedding_hnsw
ON order_reviews USING HNSW (embedding)
WITH (metric = 'cosine', ef_construction = 128, M = 16);

-- 5. Native Full-Text BM25 Index (Hybrid Lexical Search)
PRAGMA create_fts_index(
    'order_reviews',
    'review_id',
    'review_comment_title',
    'review_comment_message',
    stemmer = 'english',
    stopwords = 'english',
    lower = 1,
    overwrite = 1
);
