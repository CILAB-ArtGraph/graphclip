from .clip_run import CLIPRun
from .clip_graph_run import CLIPGraphRun
from .vit_run import ViTRun
from .run import Run
import json
from .utils import ParameterKeys
import os


def test_clip(parameters):
    test_metrics = CLIPRun(parameters).test()

    print(test_metrics)
    out_dir = parameters.get(ParameterKeys.OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/test_metrics.json", "w+") as f:
        json.dump(test_metrics, f)


def launch_experiment(parameters, run_cls: Run):
    run = run_cls(parameters)
    test_metrics = run.launch()
    print(test_metrics)
    out_dir = parameters.get(ParameterKeys.OUT_DIR)
    with open(f"{out_dir}/test_metrics.json", "w+") as f:
        json.dump(test_metrics, f)


def run_experiment(parameters, graph: bool = False, ablation: bool = False):
    if ablation:
        run_cls = ViTRun
    elif not graph:
        run_cls = CLIPGraphRun
    else:
        run_cls = CLIPRun
    launch_experiment(parameters=parameters, run_cls=run_cls)
