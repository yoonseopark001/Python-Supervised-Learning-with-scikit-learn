02

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(iris['data'], iris['target'])

=======================================================

## Predicting on unlabeled data
prediction = knn.predict(X_new)
X_new.shape

print('Prediction {}'.format(prediction))

=======================================================
