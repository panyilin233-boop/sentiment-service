import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups

# 使用 20newsgroups 的两个类别模拟情感数据
categories = ['rec.sport.baseball', 'sci.space']
data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
X_train = data.data
y_train = data.target

# 构建 pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
joblib.dump(model, 'sentiment_model.pkl')
print("模型已保存为 sentiment_model.pkl")
