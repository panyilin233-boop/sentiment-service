from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

texts = [
    "I love this movie",
    "This film is amazing",
    "What a great experience",
    "Absolutely fantastic and wonderful",
    "I really liked it",
    "This is the best thing ever",
    "I hate this movie",
    "This film is terrible",
    "What a bad experience",
    "Absolutely awful and boring",
    "I really disliked it",
    "This is the worst thing ever",
    "The product is excellent",
    "Very happy with the result",
    "Superb quality and service",
    "The product is horrible",
    "Very disappointed with the result",
    "Terrible quality and service"
]

labels = [
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0,
    1, 1, 1,
    0, 0, 0
]

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

model.fit(texts, labels)

joblib.dump(model, "sentiment_model.pkl")
print("Model saved as sentiment_model.pkl")
