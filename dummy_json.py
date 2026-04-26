import requests
import pandas as pd

# Fetch all products (limit ensures we get the full dataset)
url = "https://dummyjson.com/products?limit=194"
response = requests.get(url)

data = response.json()
products = data.get("products", [])
# Convert to DataFrame
df = pd.DataFrame(products)
print("Columns:", df.columns.tolist())

# print(df.sample(2))
# print("\nFirst product images list:")
# print(df.loc[0, "images"])

# Display full list of image URLs for the first product
# pd.set_option("display.max_colwidth", None)  # prevents truncation
# print("Full image links for first product:\n")
# print(df.loc[0, "images"])

# Explore tags column
print("Tags for first product:", df.loc[0, "tags"])

# Explore meta column
print("\nMeta info for first product:")
print("META :", df.loc[0, "meta"])

