import os
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, SpatialDropout1D, Embedding
from keras.utils.np_utils import to_categorical

# Load the saved model
model = load_model('lstm.h5')

# Load the stop words
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))

# Load the CountVectorizer
cv = CountVectorizer(max_features=75)

# Example text to predict sentiment
example = " you might not get ya bitch back "

# Apply preprocessing to the example
example = example.lower()
punctuation_signs = list("?:!.,;")
example = example
for punct_sign in punctuation_signs:
    example = example.replace(punct_sign, '')
example = example.replace('\n', ' ')
example = example.replace('\t', ' ')
example = example.replace("    ", " ")
example = example.replace('"', '')
example = example.replace("'s", "")
for stop_word in stop_words:
    regex_stopword = r"\b" + stop_word + r"\b"
    example = example.replace(regex_stopword, '')

# Convert the example to a numerical feature vector
example = cv.transform([example]).toarray()

# Make the prediction
pred = model.predict(example)
sentiment = np.argmax(pred)

# Print the predicted sentiment
if sentiment == 0:
    print("Hate Speech")
elif sentiment == 1:
    print("Offensive Speech")
else:
    print("Neither")
