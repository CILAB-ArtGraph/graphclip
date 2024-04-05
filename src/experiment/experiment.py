from .clip_run import CLIPRun
import json
from .utils import ParameterKeys
import os

def test_clip(parameters):
    test_metrics = CLIPRun(parameters).test()
    
    print(test_metrics)
    exit()
    out_dir = parameters.get(ParameterKeys.OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    
    with open(f"{out_dir}/test_metrics.json", "w+") as f:
        json.dump(test_metrics, f)