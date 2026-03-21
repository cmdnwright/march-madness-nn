import yaml

def load_config(path: str = "configs/baseline.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)