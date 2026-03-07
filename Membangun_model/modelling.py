import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Churn_Experiment")

train = pd.read_csv("train_preprocessed.csv")
test = pd.read_csv("test_preprocessed.csv")

X_train = train.drop("Exited", axis=1)
y_train = train["Exited"]

X_test = test.drop("Exited", axis=1)
y_test = test["Exited"]

mlflow.set_experiment("Churn_Model")

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)

    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # log model
    mlflow.sklearn.log_model(model, "random_forest_model")

print("Training selesai")
print("Accuracy:", acc)