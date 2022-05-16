from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier

iris_data = datasets.load_iris()
X = iris_data.data
Y = iris_data.target

model_parameters = {
    'n_estimators': [50, 150, 250],
    'max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
    'min_samples_split': [2, 4, 6]
}

rfmodel = RandomForestClassifier()

clf = GridSearchCV(rfmodel, model_parameters, cv=5)

model = clf.fit(X, Y)

pprint(model.best_estimator_.get_params())

predictions = model.predict(X)
pprint(predictions)