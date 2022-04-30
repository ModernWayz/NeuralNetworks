# Imports
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier

# Laad het iris dataset in van sklearn
iris_data = datasets.load_iris()
# Data en target uit de dataset halen als X & Y
X = iris_data.data
Y = iris_data.target

# Dictionary aanmaken met alle mogelijke hyperparameters voor de gridsearch
model_parameters = {
    'n_estimators': [50, 150, 250],
    'max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
    'min_samples_split': [2, 4, 6]
}

# Random Forest Classifier model aanmaken
rfmodel = RandomForestClassifier()

# Grid meta-estimator aanmaken
clf = GridSearchCV(rfmodel, model_parameters, cv=5)

# Train de grid meta-estimator voor het beste model te vinden
model = clf.fit(X, Y)

# Print de beste (gevonden) hyperparameters
pprint(model.best_estimator_.get_params())

# Predict op het model met de beste (gevonden) hyperparameters
predictions = model.predict(X)
print(predictions)