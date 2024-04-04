from ruamel.yaml import YAML
from pathlib import Path


def load_ruamel(path:str, typ:str="safe") -> dict:
    yaml = YAML(typ=typ)
    return yaml.load(Path(path))
