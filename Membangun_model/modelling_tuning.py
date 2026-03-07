import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("train_preprocessed.csv")
test = pd.read_csv("test_preprocessed.csv")

X_train = train.drop("Exited", axis=1)
y_train = train["Exited"]

X_test = test.drop("Exited", axis=1)
y_test = test["Exited"]


model = RandomForestClassifier()
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10]
}

grid = GridSearchCV(model, param_grid, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Best Params:", grid.best_params_)
print("Accuracy:", accuracy)