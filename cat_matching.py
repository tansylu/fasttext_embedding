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
filtered = pd.read_excel('filtered.xlsx', usecols=['Subcategory', 'Parent Category'])

# Data Cleaning for new product titles
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

filtered['Subcategory'] = filtered['Subcategory'].apply(clean_text)
filtered['Parent Category'] = filtered['Parent Category'].apply(clean_text)

# Combine "Subcategory" and "Parent Category" columns to create a single text context for new products
new_product_titles = filtered.apply(lambda row: f"{row['Subcategory']} {row['Parent Category']}", axis=1)

# Create semantic context for new product titles
new_product_contexts = embed(new_product_titles.tolist())

# Find matches for incoming new products based on cosine similarity
matches = []
for new_title, new_context in zip(new_product_titles, new_product_contexts):
    similarities = cosine_similarity([new_context], product_contexts)
    best_match_index = similarities.argmax()
    best_match_row = data.loc[best_match_index]

    matches.append((
        new_title,
        best_match_row['Product Title'],
        best_match_row['L1'],
        best_match_row['L2'],
        best_match_row['L3'],
        best_match_row['L4'],
        best_match_row['L5'],
        best_match_row['L6']
    ))

# Create a DataFrame to store the results
results_df = pd.DataFrame(matches, columns=['Subcategory', 'Matched_product_from_codex', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6'])

# Save the results to a new Excel file
results_df.to_excel('predictions.xlsx', index=False)
