import pandas as pd
from utils.text_preprocessor import transform_text

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
import joblib

import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

    # Load Dataset
df = pd.read_csv("dataset/spam.csv", encoding="latin-1")

    # Remove unnecessary columns
df = df[['v1', 'v2']]

    # Rename columns
df.columns = ['label', 'message']
    # Encode labels
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])
print("Classes:", encoder.classes_)

    # Apply preprocessing
df['processed_message'] = df['message'].apply(transform_text)

    # Show first rows
print(df.head())
print("\nShape of Dataset:")
print(df.shape)
print("\nSpam/Ham Count:")
print(df['label'].value_counts())
print("\nMissing Values:")
print(df.isnull().sum())
    # TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['processed_message']).toarray()
y = df['label']
print("TF-IDF Shape:", X.shape)
print("Labels Shape:", y.shape)
# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  
)
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
# Accuracy
print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred))
# Save Model
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(tfidf, "model/vectorizer.pkl")
print("\nModel Saved Successfully!")

# Manual Testing
test_messages = [
    "Hey bro, where are you?",
    "Congratulations! You have won a free iPhone. Claim now.",
    "URGENT! You have won ₹50000 cash. Click here.",
    "Let's meet at 6 PM."
]

for msg in test_messages:
    processed = transform_text(msg)
    vector = tfidf.transform([processed])
    pred = model.predict(vector)[0]

    print(f"\nMessage: {msg}")
    print("Prediction:", "Spam" if pred == 1 else "Ham")