from ruamel.yaml import YAML
from pathlib import Path
from safetensors import safe_open


def load_ruamel(path:str, typ:str="safe") -> dict:
    yaml = YAML(typ=typ)
    return yaml.load(Path(path))

def load_tensor(file, key, device='cpu', framework='pt'):
    with safe_open(file, device=device, framework=framework) as f:
        tensor = f.get_tensor(key)
    return tensor