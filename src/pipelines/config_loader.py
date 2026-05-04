import yaml
import os
from dotenv import load_dotenv

load_dotenv()

def load_config():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def get_env_variable(key: str):
    return os.getenv(key)