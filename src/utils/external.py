from ruamel.yaml import YAML
from pathlib import Path


def load_ruamel(path, typ="safe"):
    yaml = YAML(typ=typ)
    return yaml.load(Path(path))
