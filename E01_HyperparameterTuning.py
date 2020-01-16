# Hyperparameter Tuning - essential to use cross-validation

# GridSearchCV in scikit-learn

from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arrange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)
knn_cv.best_params_

knn_cv.best_score_

