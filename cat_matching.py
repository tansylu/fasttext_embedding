import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords

# Load the data from the Excel file (Make sure to have openpyxl installed for .xlsx files)
data = pd.read_excel('codex.xlsx')

# Load the Universal Sentence Encoder model from TensorFlow Hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Combine L1-L6 categories and product titles to create a single text context
data['Context'] = data.apply(lambda row: f"{row['L1']} {row['L2']} {row['L3']} {row['L4']} {row['L5']} {row['L6']} {row['Product Title']}", axis=1)

# Create semantic context for each product
product_contexts = embed(data['Context'].tolist())

# Read new product titles from the "filtered.xlsx" file columns "Subcategory" and "Parent Category"
filtered = pd.read_excel('food_related_emissions.xlsx', usecols=['ProductTypeName_of_hiot', 'ProductTypeName', 'CountryCode', 'CarbonFootprint','unit'])

# Data Cleaning for new product titles
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

filtered['ProductTypeName_of_hiot'] = filtered['ProductTypeName_of_hiot'].apply(clean_text)
filtered['ProductTypeName'] = filtered['ProductTypeName'].apply(clean_text)

# Combine "Subcategory" and "Parent Category" columns to create a single text context for new products
new_product_titles = filtered.apply(lambda row: f"{row['ProductTypeName_of_hiot']} {row['ProductTypeName']}", axis=1)

# Create semantic context for new product titles
new_product_contexts = embed(new_product_titles.tolist())

# Find matches for each product in the "codex.xlsx" file based on cosine similarity
matches = []

for codex_title, codex_context in zip(data['Product Title'], product_contexts):
    similarities = cosine_similarity([codex_context], new_product_contexts)[0]
    best_match_index = similarities.argmax()
    best_match_row = filtered.loc[best_match_index]

    matches.append((
        codex_title,
        best_match_row['ProductTypeName_of_hiot'],
        best_match_row['ProductTypeName'],
         best_match_row['CarbonFootprint'],
         best_match_row['unit'],
         best_match_row['CountryCode']
    ))

# Create a DataFrame to store the results
results_df = pd.DataFrame(matches, columns=['Codex_product_title', 'Matched_ProductTypeName_of_hiot', 'Matched_ProductTypeName_of_hiot','CarbonFootprint', 'unit', 'COuntry COde'])

# Save the results to a new Excel file
results_df.to_excel('codex_matches.xlsx', index=False)
