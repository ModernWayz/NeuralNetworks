from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

iris_data = datasets.load_iris()
X = iris_data.data
Y = iris_data.target

model_parameters = {
    'n_estimators': randint(0, 300),
    'max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
    'min_samples_split': uniform(0, 1)
}

rfmodel = RandomForestClassifier()

clf = RandomizedSearchCV(rfmodel, model_parameters, n_iter = 100, cv = 5, random_state = 1)

model = clf.fit(X, Y)

pprint(model.best_estimator_.get_params())

predictions = model.predict(X)
pprint(predictions)