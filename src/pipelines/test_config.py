from config_loader import load_config, get_env_variable

config = load_config()

print("Test Size:", config["data"]["test_size"])
print("Model Type:", config["model"]["type"])

print("Model Name (env):", get_env_variable("MODEL_NAME"))
print("MLflow URI:", get_env_variable("MLFLOW_TRACKING_URI"))