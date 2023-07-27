import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, MWETokenizer
import fasttext.util
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity

# Download the necessary resources for NLTK
nltk.download('punkt')

def get_word_embeddings(text, model):
    # Tokenize the text into individual words and multi-word expressions (phrases)
    words = word_tokenize(text.lower())
    phrases = [" ".join(phrase) for phrase in MWETokenizer().tokenize(words)]
    
    embeddings = [model.wv[word] for word in phrases if word in model.wv]
    return embeddings

def get_average_embedding(embeddings):
    if not embeddings:
        return None
    return np.mean(embeddings, axis=0)

def create_embeddings_for_table(data, model):
    embeddings_dict = {}
    for _, row in data.iterrows():
        product = row['Subcategory']
        category = row['Parent Category']
        
        product_embeddings = get_word_embeddings(product, model)
        category_embeddings = get_word_embeddings(category, model)

        product_average_embedding = get_average_embedding(product_embeddings)
        category_average_embedding = get_average_embedding(category_embeddings)

        embeddings_dict[product] = product_average_embedding
        embeddings_dict[category] = category_average_embedding
    
    return embeddings_dict

def find_most_similar_product(query_product_embedding, embeddings_dict, threshold=0.5):
    best_match_product = None
    best_similarity_score = -1

    for product, product_embedding in embeddings_dict.items():
        similarity_score = cosine_similarity([query_product_embedding], [product_embedding])[0][0]
        if similarity_score > best_similarity_score and similarity_score >= threshold:
            best_similarity_score = similarity_score
            best_match_product = product

    return best_match_product

# Load data from Excel sheet and create DataFrames
filtered_data = pd.read_excel("filtered.xlsx")
codex_data = pd.read_excel("codex.xlsx")

# Create a FastText model and train it on the "filtered.xlsx" data
model = FastText(vector_size=300, window=5, min_count=1, workers=4)
sentences = [word_tokenize(product.lower()) for product in filtered_data['Subcategory']]
model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=30)

# Create embeddings for the filtered data
filtered_embeddings_dict = create_embeddings_for_table(filtered_data, model)

# Find similar products for each item in the "Product" column of the codex data and display their categories
results = []
for product in codex_data['Product']:
    product_embeddings = get_word_embeddings(product, model)
    query_product_embedding = get_average_embedding(product_embeddings)
    best_match_product = find_most_similar_product(query_product_embedding, filtered_embeddings_dict)
    
    if best_match_product is not None:
        category = filtered_data.loc[filtered_data['Subcategory'] == best_match_product, 'Parent Category'].values
        if len(category) > 0:
            category = category[0]
        else:
            category = "Not found"
        results.append([product, best_match_product, category])
    else:
        results.append([product, "Not found", "Not found"])

# Create a new DataFrame for the results
output_columns = ['Query Product', 'Similar Product', 'Category']
output_df = pd.DataFrame(results, columns=output_columns)

# Save the results to a new Excel file
output_df.to_excel("output.xlsx", index=False)
