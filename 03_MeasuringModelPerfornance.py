
## Train/Test split: X_train, X_text, y_train, y_test

from sklearn.model_selection import train_test_split

X_train, X_text, y_train, y_test =     # as a default
  train_test_split(X, y, test_size=0.3,
                   random_state=21,    # random seed
                   stratify=y)         # list or array
                   
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
