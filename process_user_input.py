import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re

# Data Cleaning for new product titles
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def match(input):
    input = clean_text(input)
    # Load the Universal Sentence Encoder model from TensorFlow Hub
    embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    ref_db = np.load('ref_db.npy')
    value = embedder([input])
    similar = cosine_similarity(ref_db, value)
    best_match_index = similar.argmax()
    print(best_match_index, " best_match_index")
    # Load the data from the Excel file
    filtered = pd.read_excel('food_related_emissions.xlsx', usecols=['ProductTypeName_of_hiot', 'ProductTypeName', 'CountryCode', 'CarbonFootprint','unit'])
    best_match_row = filtered.loc[best_match_index]
      # Extract individual outputs
    product_type_name = best_match_row['ProductTypeName_of_hiot']
    product_type = best_match_row['ProductTypeName']
    carbon_footprint = best_match_row['CarbonFootprint']
    unit = best_match_row['unit']
    country_code = best_match_row['CountryCode']
    
    return product_type_name, product_type, carbon_footprint, unit, country_code