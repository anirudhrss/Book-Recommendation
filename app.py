
    
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def get_recommendations(authors, title):
    # Load the necessary data
    finaldf = pd.read_csv(r"C:\Users\aniru\Music\Zally\finaldf.csv")
    df_main = pd.read_csv(r"C:\Users\aniru\Music\Zally\df_main.csv")

    newdf = pd.DataFrame()
    data = {'authors': [authors], 'title': [title]}
    newdf = pd.DataFrame(data)

    vectorizer4 = TfidfVectorizer()
    authornew_tfidf = vectorizer4.fit_transform(newdf['authors'])

    vectorizer5 = TfidfVectorizer()
    titlenew_tfidf = vectorizer5.fit_transform(newdf['title'])

    authornewscoredf = pd.DataFrame(authornew_tfidf.toarray(), columns=vectorizer4.get_feature_names_out())
    titlenewscoredf = pd.DataFrame(titlenew_tfidf.toarray(), columns=vectorizer5.get_feature_names_out())

    newdf['authornew_tfidf'] = authornewscoredf.sum(axis=1)
    newdf['titlenew_tfidf'] = titlenewscoredf.sum(axis=1)

    finaldf1 = newdf.drop(columns=['authors', 'title'])

    cos_sim = cosine_similarity(finaldf[['author_tfidf', 'title_tfidf']], finaldf1)
    cosine_sim_df = pd.DataFrame(cos_sim)

    # Get top 5 rows with highest cosine similarity scores
    top_5_rows = cosine_sim_df.stack().nlargest(5).reset_index()
    top_5_rows.columns = ['row1', 'row2', 'similarity']

    # Get text values of top 5 rows
    top_5_text = df_main.iloc[top_5_rows['row1'].values][['title', 'authors']].values

    return top_5_text, top_5_rows['similarity']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    authors = request.form['authors']
    title = request.form['title']

    top_5_text, similarity_scores = get_recommendations(authors, title)

    return render_template('recommendations.html', recommendations=top_5_text, similarity=similarity_scores)

if __name__ == '__main__':
    app.run(debug=True, port = 8080)