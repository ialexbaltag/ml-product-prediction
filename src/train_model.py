import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import joblib


df = pd.read_csv("data/products.csv")

# stadardized colmns titles
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace("*", "", regex=False)
    .str.replace(" ", "_", regex=False)
    .str.strip("_")
)


# Drop all rows with missing values
df = df.dropna()

#standadized category_label 
df["category_label_cleaned"] = (df["category_label"].astype(str).str.strip())

df = df.drop(columns=['category_label_cleaned'])

# Drop any rows where conversion failed (still NaN)
df = df.dropna(subset=['category_label'])


# Same operation for column product_title
df["product_title_cleaned"] = (df["product_title"].astype(str).str.strip())

df = df.drop(columns=['product_title_cleaned'])

# Drop any rows where conversion failed (still NaN)
df = df.dropna(subset=['product_title'])

# Convert all category label values to lowercase and strip extra spaces
df['category_label'] = df['category_label'].astype(str).str.lower().str.strip()

# Convert column type to 'category'
df['category_label'] = df['category_label'].astype('category')


# Drop columns that are not useful for modeling
df = df.drop(columns=['product_id', 'merchant_id','number_of_views', 'merchant_rating','listing_date'])

# Create new column with length of each product_title
df['product_title_length'] = df['product_title'].astype(str).str.len()

# Extract the first word from product_title
df["product_brand"] = (
    df["product_title"]
    .fillna("")
    .str.strip()
    .str.split()
    .str[0]
    .str.lower()
)

# training models

# Features and label
X = df[["product_brand", "product_title", "product_title_length"]]
y = df["category_label"]


# Preprocessor: TF-IDF for text, MinMaxScaler for numeric feature
preprocessor = ColumnTransformer(
    transformers=[
        ("brand", TfidfVectorizer(), "product_brand"),
        ("title", TfidfVectorizer(), "product_title"),
        ("length", MinMaxScaler(), ["product_title_length"])
    ]
)

# used RandomForestClassifier
pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier())
])

# train the model on the entire dataset
pipeline.fit(X, y)


# save the model to a file
joblib.dump(pipeline,"model/category_model.pkl")

print("Model trained and saved as 'model/category_model.pkl'")