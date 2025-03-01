"""
Utility to load configuration settings.
"""

import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    print(cfg)
