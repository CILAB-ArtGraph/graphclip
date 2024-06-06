from .clip_run import CLIPRun
from .clip_graph_run import CLIPGraphRun
from .vit_run import ViTRun
from .vit_multitask_run import ViTMultiTaskRun
from .run import Run
from .clip_multitask_run import CLIPMultitaskRun
from .clip_graph_multitask_run import CLIPGraphMultitaskRun
import json
from .utils import ParameterKeys
import os


def launch_test(parameters, run_cls: Run):
    run = run_cls(parameters)
    test_metrics = run.test()
    print(test_metrics)
    out_dir = parameters.get(ParameterKeys.OUT_DIR)
    with open(f"{out_dir}/test_metrics.json", "w+") as f:
        json.dump(test_metrics, f)


def test_single_task(parameters, ablation, graph):
    if ablation:
        run_cls = ViTRun
    elif graph:
        run_cls = CLIPGraphRun
    else:
        run_cls = CLIPRun
    launch_test(parameters=parameters, run_cls=run_cls)


def test_multitask(parameters, ablation, graph):
    if ablation:
        run_cls = ViTMultiTaskRun
    elif graph:
        run_cls = CLIPGraphMultitaskRun
    else:
        run_cls = CLIPMultitaskRun
    launch_test(parameters=parameters, run_cls=run_cls)


def run_test(
    parameters, graph: bool = False, ablation: bool = False, multitask: bool = False
):
    if multitask:
        test_multitask(parameters=parameters, graph=graph, ablation=ablation)
    else:
        test_single_task(parameters=parameters, graph=graph, ablation=ablation)


def launch_experiment(parameters, run_cls: Run):
    run = run_cls(parameters)
    test_metrics = run.launch()
    print(test_metrics)
    out_dir = parameters.get(ParameterKeys.OUT_DIR)
    with open(f"{out_dir}/test_metrics.json", "w+") as f:
        json.dump(test_metrics, f)


def launch_multitask(parameters, graph: bool = False, ablation: bool = False):
    if ablation:
        run_cls = ViTMultiTaskRun
    elif graph:
        run_cls = CLIPGraphMultitaskRun
    else:
        run_cls = CLIPMultitaskRun
    launch_experiment(parameters=parameters, run_cls=run_cls)


def launch_single_task(parameters, graph: bool = False, ablation: bool = False):
    if ablation:
        run_cls = ViTRun
    elif graph:
        run_cls = CLIPGraphRun
    else:
        run_cls = CLIPRun
    launch_experiment(parameters=parameters, run_cls=run_cls)


def run_experiment(
    parameters, graph: bool = False, ablation: bool = False, multitask: bool = False
):
    if multitask:
        launch_multitask(parameters=parameters, graph=graph, ablation=ablation)
    else:
        launch_single_task(parameters=parameters, graph=graph, ablation=ablation)
