import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import nltk
import re
from re import sub
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score

stopwords = (stopwords.words("english"))
stemmer = nltk.PorterStemmer()
data = pd.read_csv("labeled_data.csv")

data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Speech", 2: "Neither"})
data = data[["tweet", "labels"]]


def cleaning_text(text):
    text = str(text).lower()
    text = sub('[.?]', '', text)
    text = sub('https?://\S+|www.\S+', '', text)
    text = sub('<.?>+', '', text)
    text = sub(r'[^\w\s]', '', text)
    text = sub('/n', '', text)
    text = sub('\w\d\w', '', text)
    text = [word for word in text.split(" ") if word not in stopwords]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(" ")]
    text = " ".join(text)
    return text


data["tweet"] = data["tweet"].apply(cleaning_text)

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()

X = cv.fit_transform(x)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test,y_pred))

example = "nigga you are so dark that we have to search for you even in the daylight"
example = cv.transform([example]).toarray()
print(model.predict(example))
