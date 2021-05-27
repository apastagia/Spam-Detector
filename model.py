import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.drop(['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'], axis = 1)
df['label'] = df['v1'].map({'ham':0, 'spam':1})

X = df['v2']
y = df['v1']

#extract features with countvectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

pickle.dump(cv, open('transform.pkl','wb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename,'wb'))