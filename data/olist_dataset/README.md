# Olist Brazilian E-Commerce Dataset

To populate this directory with the raw source CSV files:
1. Download the Brazilian E-Commerce Public Dataset by Olist from Kaggle:
   https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
2. Extract the CSV files into this directory (`data/olist_dataset/`):
   - `olist_customers_dataset.csv`
   - `olist_geolocation_dataset.csv`
   - `olist_orders_dataset.csv`
   - `olist_order_items_dataset.csv`
   - `olist_order_payments_dataset.csv`
   - `olist_order_reviews_dataset.csv`
   - `olist_products_dataset.csv`
   - `olist_sellers_dataset.csv`
   - `product_category_name_translation.csv`
3. Run the ingestion pipeline to compile `olist.duckdb`:
   ```bash
   uv run python -m dualquery.ingestion.olist_loader
   ```
