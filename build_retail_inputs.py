from pathlib import Path
import pandas as pd

base_dir = Path(r"c:\Users\Ebi_Mahmdli\Desktop")
source_xlsx = base_dir / "online_retail_data" / "Online Retail.xlsx"

if not source_xlsx.exists():
    raise FileNotFoundError(f"Source file not found: {source_xlsx}")

df = pd.read_excel(source_xlsx)
df = df[["InvoiceNo", "Description", "Quantity"]].copy()

df = df.dropna(subset=["InvoiceNo", "Description"])
df["Description"] = df["Description"].astype(str).str.strip()
df = df[df["Description"] != ""]

df = df[df["Quantity"] > 0]
df = df[~df["InvoiceNo"].astype(str).str.startswith("C", na=False)]

# Keep top products to keep matrix size practical
product_counts = df["Description"].value_counts()
top_n = 300
selected_products = product_counts.head(top_n).index

df = df[df["Description"].isin(selected_products)]

df_products = pd.DataFrame({"product_name": sorted(selected_products)})
product_names = df_products["product_name"].tolist()

basket = (
    df.groupby(["InvoiceNo", "Description"]).size().unstack(fill_value=0)
)

basket = basket.reindex(columns=product_names, fill_value=0)
basket = (basket > 0).astype(int)

cooccurrence = basket.T.dot(basket)
cooccurrence = cooccurrence.loc[product_names, product_names]

products_path = base_dir / "products.csv"
cooccurrence_path = base_dir / "product_cooccurrence.csv"

df_products.to_csv(products_path, index=False)
cooccurrence.to_csv(cooccurrence_path)

print(f"Saved: {products_path}")
print(f"Saved: {cooccurrence_path}")
print(f"Products count: {len(df_products)}")
print(f"Cooccurrence shape: {cooccurrence.shape}")
