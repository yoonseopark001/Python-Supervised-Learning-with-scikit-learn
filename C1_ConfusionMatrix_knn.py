# Confusion matrix in scikit-learn

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

knn = KNeighborClassifier(n_neighbors=8)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.4, random_state=42)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

# accuracy = (tp+tn)/(tp+tn+fp+fn)
# precision = tp / (tp+fp)   ; eg. not mant real emails predicted as spam
# recall = tp / (tp+fn)      ; eg. predicted most spam emails correctly
# f1-score = 2*(precision*recall)/(precision+recall)
# support

(# reference: http://it.plusblog.co.kr/221243790904)
