from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = open('Movies_TV.txt').read()
import re
corpus = re.sub(r'Domain.*\n', '', corpus)
rows = corpus.split('\n')
rows.remove(rows[-1])

inputData, y = [], []
for row in rows:
    _, label, _, review = row.split('\t')
    inputData.append(review)
    y.append(label)
    
    
#unigram +Binary Based
print("UniGram")
vec = CountVectorizer(ngram_range = (1, 1), max_features = 1000,binary=True,max_df=100,min_df=10)
X = vec.fit_transform(inputData)
print(X.toarray())
print(vec.vocabulary_)
#Bigram +Binary Based
print("Bigram")
vec = CountVectorizer(ngram_range = (1, 2), max_features = 1000,binary=True,max_df=100,min_df=10)
X = vec.fit_transform(inputData)
print(X.toarray())
print(vec.vocabulary_)
#Trigram +Binary Based
print("Trigram")
vec = CountVectorizer(ngram_range = (1, 3), max_features = 1000,binary=True,max_df=100,min_df=10)
X = vec.fit_transform(inputData)
print(X.toarray())
print(vec.vocabulary_)

#unigram +Frequency Based
print("UniGram")
vec = CountVectorizer(ngram_range = (1, 1), max_features = 1000,max_df=100,min_df=10)
X = vec.fit_transform(inputData)
print(X.toarray())
print(vec.vocabulary_)
#Bigram +Frequency Based
print("Bigram")
vec = CountVectorizer(ngram_range = (1, 2), max_features = 1000,max_df=100,min_df=10)
X = vec.fit_transform(inputData)
print(X.toarray())
print(vec.vocabulary_)
#Trigram +Frequency  Based
print("Trigram")
vec = CountVectorizer(ngram_range = (1, 3), max_features = 1000,max_df=100,min_df=10)
X = vec.fit_transform(inputData)
print(X.toarray())
print(vec.vocabulary_)

#Unigram +Tfidf
print("UniGram")
vec = CountVectorizer(ngram_range = (1, 1), max_features = 1000,max_df=100,min_df=10)
matrix_X = tfidf.fit_transform(X)
matrix_X.toarray()
print(X.toarray())
print(vec.vocabulary_)
#Bigram +Tfidf
print("UniGram")
vec = CountVectorizer(ngram_range = (1, 2), max_features = 1000,max_df=100,min_df=10)
matrix_X = tfidf.fit_transform(X)
matrix_X.toarray()
print(X.toarray())
print(vec.vocabulary_)
#Trigram +Tfidf
print("UniGram")
vec = CountVectorizer(ngram_range = (1, 3), max_features = 1000,max_df=100,min_df=10)
matrix_X = tfidf.fit_transform(X)
matrix_X.toarray()
print(X.toarray())
print(vec.vocabulary_)