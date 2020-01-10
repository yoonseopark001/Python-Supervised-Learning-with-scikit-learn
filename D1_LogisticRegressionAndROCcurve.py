# Logistic Regression(logreg) is used for classification problems (not Regression problems!)

# Logistic regression in scikit-learn

from sklearn.linear_model import logisticRegression
from sklearn.model_selection import train_test_split

logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# plotting the ROC curve (the receiver operating characteristic curve)

from sklearn.metrics import roc_curve
y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob) # fpr : false positive rate # tpr : true positive rate # thresholds 

plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()


# Probability thresholds
#  by default, logistic regression threshold = 0.5

