import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    features = ['Product Name', 'Category', 'Product Specification']
    df['combined_features'] = df[features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return df, cosine_sim

# Search function
def search_products(df, query):
    results = df[df['Product Name'].str.contains(query, case=False)]
    return results

# Recommendation function
def get_recommendations(product_id, cosine_sim, df, top_n=5):
    idx = df[df['Uniq Id'] == product_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    product_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[product_indices][['Uniq Id', 'Product Name', 'Category', 'Selling Price', 'Product Specification', 'Image', 'Product Url']]
    return recommendations

# Load the CSV file (Replace 'new_dataset.csv' with your actual file path)
data, cosine_sim = load_data('new_dataset.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        search_query = request.form['search_query']
        filtered_data = search_products(data, search_query)
        search_results = []

        if len(filtered_data) > 0:
            for _, row in filtered_data.iterrows():
                product_name = row['Product Name']
                unique_id = row['Uniq Id']
                product_url = row['Product Url']

                recommendations = get_recommendations(unique_id, cosine_sim, data)
                recommendations_html = recommendations[['Product Name', 'Uniq Id', 'Product Url']].to_html(index=False, escape=False, render_links=True)

                search_results.append({
                    'product_name': product_name,
                    'unique_id': unique_id,
                    'product_url': product_url,
                    'recommendations': recommendations_html
                })
        else:
            search_results = []

        return render_template('index.html', search_results=search_results, search_query=search_query)

    return render_template('index.html', search_results=[], search_query='')

if __name__ == '__main__':
    app.run(debug=True)