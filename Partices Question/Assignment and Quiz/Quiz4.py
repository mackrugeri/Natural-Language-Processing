import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

corpus = pd.read_csv('IMDB Dataset.csv',encoding="UTF-8", engine='c')
reviews = []
labels = []
for i in corpus.review:
    reviews.append(i)
reviews = reviews[:3000]

vec = TfidfVectorizer()
matrix_X = vec.fit_transform(reviews)
LDA = LatentDirichletAllocation(n_components=10)
LDA.fit(matrix_X)
feature = vec.get_feature_names()
for id, topic in enumerate(LDA.components_):
  print(id)
  print("WordID List: ", topic.argsort()[:-16:-1])
  print("Word List: ",[feature[j] for j in topic.argsort()[:-16:-1]])

  LDA = LatentDirichletAllocation(n_components=15)
  LDA.fit(matrix_X)
  feature = vec.get_feature_names()
  for id, topic in enumerate(LDA.components_):
    print(id)
    print("WordID List: ", topic.argsort()[:-16:-1])
    print("Word List: ",[feature[j] for j in topic.argsort()[:-16:-1]])


    LDA = LatentDirichletAllocation(n_components=20)
    LDA.fit(matrix_X)
    feature = vec.get_feature_names()
    for id, topic in enumerate(LDA.components_):
      print(id)
      print("WordID List: ", topic.argsort()[:-16:-1])
      print("Word List: ",[feature[j] for j in topic.argsort()[:-16:-1]])