import click
from src.utils import load_ruamel


@click.group()
def main():
    pass


@main.command("split")
@click.option("--parameters", help="Path to parameters file")
@click.option("--graph", is_flag=True, default=False)
@click.option("--classes", is_flag=True, default=False)
def split(parameters, graph, classes):
    from src.preprocess import split as split_fn

    split_fn(parameters=load_ruamel(parameters), graph=graph, classes=classes)


@main.command("download")
@click.option("--parameters", help="Path to parameters file")
def download(parameters):
    from src.preprocess import download_artgraph

    download_artgraph(load_ruamel(parameters))


@main.command("extract_features")
@click.option("--parameters", help="Path to parameters file")
@click.option("--t", help="Either visual or label")
def extract_features(parameters, t):
    from src.features import extract_features

    extract_features(load_ruamel(parameters), t)


@main.command("test")
@click.option("--parameters", help="Path to parameters file")
@click.option("--graph", is_flag=True, default=False)
@click.option("--ablation", is_flag=True, default=False)
@click.option("--multitask", is_flag=True, default=False)
def test(parameters, graph, ablation, multitask):
    from src.experiment import run_test as test_fn

    test_fn(load_ruamel(parameters), graph, ablation, multitask)


@main.command("experiment")
@click.option("--parameters", help="Path to parameters file")
@click.option("--graph", is_flag=True, default=False)
@click.option("--ablation", is_flag=True, default=False)
@click.option("--multitask", is_flag=True, default=False)
def experiment(parameters, graph, ablation, multitask):
    from src.experiment import run_experiment

    run_experiment(load_ruamel(parameters), graph, ablation, multitask)
