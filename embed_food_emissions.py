import pandas as pd
import tensorflow_hub as hub
import re

# Load the data from the Excel file
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

# Get the list of new product titles
new_product_titles_list = new_product_titles.tolist()

# Load the Universal Sentence Encoder model from TensorFlow Hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Embed the new product titles
new_product_contexts = embed(new_product_titles_list)

# Save the embedded contexts to a file
import numpy as np

# Save the embedded contexts as a NumPy array
np.save('ref_db.npy', new_product_contexts.numpy())
