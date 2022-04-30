# Imports
from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

# Laad het iris dataset in van sklearn
iris_data = datasets.load_iris()
# Data en target uit de dataset halen als X & Y
X = iris_data.data
Y = iris_data.target

# Dictionary aanmaken met alle mogelijke hyperparameters voor de random search
model_parameters = {
    'n_estimators': randint(0, 300),
    'max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
    'min_samples_split': uniform(0, 2)
}

# Random Forest Classifier model aanmaken
rfmodel = RandomForestClassifier()

# Random meta-estimator aanmaken
clf = RandomizedSearchCV(rfmodel, model_parameters, n_iter=100, cv=5, random_state=1)

# Train de random meta-estimator voor het beste model te vinden uit de 100 iteraties
model = clf.fit(X, Y)

# Print de beste (gevonden) hyperparameters
pprint(model.best_estimator_.get_params())

# Predict op het model met de beste (gevonden) hyperparameters
predictions = model.predict(X)
print(predictions)