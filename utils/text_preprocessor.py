import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    # Lowercase
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    y = []

    # Remove special characters
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    # Remove stopwords
    for word in text:
        if word not in stopwords.words('english'):
            y.append(word)

    text = y[:]
    y.clear()

    # Stemming
    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)