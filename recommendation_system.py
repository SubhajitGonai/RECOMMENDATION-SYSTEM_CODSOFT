import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data: Movies and their descriptions
data = {
    'movie': ['The Matrix', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump'],
    'description': [
        'A computer hacker learns from mysterious rebels about the true nature of his reality.',
        'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
        'When the menace known as The Joker emerges from his mysterious past, he wreaks havoc and chaos on the people of Gotham.',
        'The lives of two mob hitmen, a boxer, a gangster’s wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
        'The presidencies of Kennedy and Johnson, the Vietnam War, the Civil Rights Movement, the space race and other historical events unfold from the perspective of one man.'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['description'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(title):
    if title not in df['movie'].values:
        return "Movie not found in the dataset."

    idx = df.index[df['movie'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Get top 3 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return df['movie'].iloc[movie_indices]

# Test the function
if __name__ == "__main__":
    movie_title = 'The Matrix'
    recommendations = get_recommendations(movie_title)
    print(f"Recommendations for '{movie_title}':")
    for rec in recommendations:
        print(f"- {rec}")
