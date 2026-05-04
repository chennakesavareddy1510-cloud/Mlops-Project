import os
import mlflow
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.data.load_data import load_data
from src.pipelines.config_loader import load_config, get_env_variable


def train():
    # Load config
    config = load_config()

    mlflow_uri = get_env_variable("MLFLOW_TRACKING_URI")

    if mlflow_uri:
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("mlops_experiment")
            mlflow_enabled = True
        except Exception as e:
            print("MLflow not available, running without tracking")
            mlflow_enabled = False
    else:
        mlflow_enabled = False

    # Load data
    df = load_data()
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    # Start MLflow run
    if mlflow_enabled:
        mlflow.start_run()

        model = RandomForestClassifier(
            n_estimators=config["model"]["n_estimators"]
        )

        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        # Log params + metrics
        mlflow.log_param("n_estimators", config["model"]["n_estimators"])
        mlflow.log_metric("accuracy", accuracy)

        # Save model
        joblib.dump(model, "model.pkl")

        # Log artifact
        mlflow.log_artifact("model.pkl")

        print(f"Accuracy: {accuracy}")
        mlflow.end_run()


if __name__ == "__main__":
    train()