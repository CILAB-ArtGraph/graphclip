from .clip_run import CLIPRun
from .clip_graph_run import CLIPGraphRun
from .vit_run import ViTRun
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


def fine_tune_clip(parameters):
    run = CLIPRun(parameters=parameters)
    test_metrics = run.launch()
    print(test_metrics)
    out_dir = parameters.get(ParameterKeys.OUT_DIR)
    with open(f"{out_dir}/test_metrics.json", "w+") as f:
        json.dump(test_metrics, f)


def fine_tune_vit(parameters):
    run = ViTRun(parameters=parameters)
    test_metrics = run.launch()
    print(test_metrics)
    out_dir = parameters.get(ParameterKeys.OUT_DIR)
    with open(f"{out_dir}/test_metrics.json", "w+") as f:
        json.dump(test_metrics, f)


def run_experiment(parameters, graph: bool = False, ablation: bool = False):
    if ablation:
        fine_tune_vit(parameters)
    elif not graph:
        fine_tune_clip(parameters=parameters)
    else:
        train_clip_graph(parameters=parameters)


def train_clip_graph(parameters):
    run = CLIPGraphRun(parameters=parameters)
    test_metrics = run.launch()
    print(test_metrics)
    out_dir = parameters.get(ParameterKeys.OUT_DIR)
    with open(f"{out_dir}/test_metrics.json", "w+") as f:
        json.dump(test_metrics, f)
