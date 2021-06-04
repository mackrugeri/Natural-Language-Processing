from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
corpus = open('badges.data').read()
X, y = [], []
for row in data:
    label = row[:1]
    name = row[2:]
    X.append(name)
    y.append(label)
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(max_features = 20)
matrix_X = vec.fit_transform(X)
train_x, test_x, train_y, test_y = train_test_split(matrix_X, y, shuffle = True, train_size = 0.7)
dtc = DecisionTreeClassifier()
lc = SGDClassifier()
lc.fit(train_x, train_y)
dtc.fit(train_x, train_y)
labels1 = dtc.predict(test_x)
labels2 = lc.predict(test_x)
from sklearn.metrics import accuracy_score
print('Accuracy DTC: ', accuracy_score(test_y, labels1))
print('Accuracy LC: ', accuracy_score(test_y, labels2))