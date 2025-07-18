
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv('news.csv')  # Make sure the news.csv file is in the same directory
print("Dataset shape:", df.shape)
print("Dataset preview:")
print(df.head())

# Split data
X = df['text']
y = df['label']

# Text vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=7)

# Model training
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(score*100, 2)}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print("Confusion Matrix:\n", cm)
