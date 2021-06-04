import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score
from sklearn.linear_model import SGDClassifier
from string import punctuation as punc
import re
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import nltk
nltk.download('wordnet')
nltk.download('sentiwordnet')
corpus = pd.read_csv('IMDB Dataset.csv',encoding="UTF-8", engine='c')
reviews = []
labels = []
for i in corpus.review:
    reviews.append(i)
for i in corpus.sentiment:
    labels.append(i)
sentiment = []
for review in reviews:
    review = re.sub("<br /><br />","",review)
    review = re.sub("["+punc+"]","",review)
    wordList = re.split(" ",review)
    total_score = 0
    for word in wordList:
        syn_t = wn.synsets(word)
        if len(syn_t) > 0:
            syn_t = syn_t[0]
            s_syn_t = swn.senti_synset(syn_t.name())
            score = s_syn_t.pos_score() - s_syn_t.neg_score()
            total_score += score
    if total_score > 0:
        sentiment.append("+ve")
    elif total_score < 0:
        sentiment.append("-ve")
    else:
        sentiment.append("neutral")
vec = CountVectorizer(min_df=10, lowercase=True)
matrix_X = vec.fit_transform(reviews)
train_x , test_x , train_y , test_y = train_test_split(matrix_X, labels, shuffle=True, train_size = 0.7)
linear = SGDClassifier()
linear.fit(train_x, train_y)
linearPredict = linear.predict(test_x)
accuracy = accuracy_score(test_y, linearPredict)
print("Accuracy of Labels DecisionTreeClassifier", accuracy)

vec = CountVectorizer(min_df=10, lowercase=True)
matrix_X = vec.fit_transform(reviews)
train_x , test_x , train_y , test_y = train_test_split(matrix_X, sentiment, shuffle=True, train_size = 0.7)
linear = SGDClassifier()
linear.fit(train_x, train_y)
linearPredict = linear.predict(test_x)
accuracy = accuracy_score(test_y, linearPredict)
