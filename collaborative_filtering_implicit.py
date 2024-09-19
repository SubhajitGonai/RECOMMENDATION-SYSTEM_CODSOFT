import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import train_test_split
from implicit.evaluation import precision_at_k

# Create a sample user-item interaction matrix
interactions = pd.DataFrame({
    'user': [0, 0, 1, 1, 2, 2, 2],
    'item': [0, 1, 1, 2, 0, 1, 2],
    'rating': [5, 3, 4, 2, 1, 5, 2]
})

user_item_matrix = interactions.pivot(index='user', columns='item', values='rating').fillna(0)
user_item_matrix = user_item_matrix.values

# Train-test split
train, test = train_test_split(user_item_matrix, train_percentage=0.8)

# Train model
model = AlternatingLeastSquares(factors=10, regularization=0.1, iterations=20)
model.fit(train)

# Evaluate model
precision = precision_at_k(model, test, k=3)
print(f'Precision at k=3: {precision:.2f}')

def recommend_items(user_id, top_n=3):
    scores = model.recommend(user_id, user_item_matrix[user_id], N=top_n)
    return scores

# Example recommendation
user_id = 0
recommendations = recommend_items(user_id)
print(f"Recommendations for user {user_id}:")
for rec in recommendations:
    print(f"- Item {rec[0]} with score {rec[1]:.2f}")
