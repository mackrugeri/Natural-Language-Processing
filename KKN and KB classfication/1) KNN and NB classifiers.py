from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

corpus = open('badges.data').read()
data = corpus.split('\n')
data.remove(data[-1])
data.remove(data[0])
X_label,Y_value=[],[]
for row in data:
    label = row[:1]
    value = row[2:]
    X_label.append(label)
    Y_value.append(value)
vec = CountVectorizer(max_features = 20)
matrix_Y = vec.fit_transform(Y_value)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
matrix_Y = vec.fit_transform(Y_value)
train_x,test_x,train_y,test_y = train_test_split(matrix_Y,X_label,shuffle=True,train_size = 0.7)
knnc = KNeighborsClassifier(n_neighbors = 5)
nbc = MultinomialNB()
knn.fit(train_x, train_y)
nbc.fit(train_x, train_y)
labels_knn = knn.predict(test_x)
labels_nbc = nbc.predict(test_x)
print('Accuracy of knn: ', accuracy_score(test_y, labels_knn))
print('Accuracy of nbc: ', accuracy_score(test_y, labels_nbc))