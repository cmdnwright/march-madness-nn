import yaml

def load_config(path: str = "configs/baseline.yaml") -> dict:
    '''loads a yaml config from configs directory

    Parameters
    ----------
    path : str, optional
        the full config path relative to the root, by default "configs/baseline.yaml"

    Returns
    -------
    dict
        the config
    '''
    with open(path, "r") as f:
        return yaml.safe_load(f)