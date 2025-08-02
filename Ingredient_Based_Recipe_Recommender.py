from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load known ingredients from file
def load_known_ingredients(file_path="simplified_valid_ingredients.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

known_ingredients = load_known_ingredients()

# Function to load data
def load_data():
    file_path = "D:/project_file/archana.csv"  # Adjust as needed
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['TranslatedIngredients', 'TranslatedInstructions'])
    df['TranslatedIngredients'] = df['TranslatedIngredients'].str.lower().str.strip()
    df['TranslatedInstructions'] = df['TranslatedInstructions'].str.strip()
    return df

df = load_data()

# Function to recommend recipes
def recommend_recipes(user_ingredients, top_n=50):
    # Preprocess user input
    user_ingredients_list = [ing.strip().lower() for ing in user_ingredients.split(',')]
    filtered_ingredients = [ing for ing in user_ingredients_list if ing in known_ingredients]

    if not filtered_ingredients:
        return []

    cleaned_input = ', '.join(filtered_ingredients)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['TranslatedIngredients'])

    user_query_vec = vectorizer.transform([cleaned_input])
    similarities = cosine_similarity(user_query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    if similarities[top_indices[0]] == 0:
        return []

    recommendations = df.iloc[top_indices]
    return recommendations[['TranslatedRecipeName', 'TranslatedIngredients', 'TranslatedInstructions', 'TotalTimeInMins', 'URL']].to_dict(orient="records")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    ingredients = request.form.get("ingredients", "")
    recommendations = recommend_recipes(ingredients)
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
