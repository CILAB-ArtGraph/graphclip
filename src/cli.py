import click
from src.utils import load_ruamel


@click.group()
def main():
    pass


@main.command("split")
@click.option("--parameters", help="Path to parameters file")
def split(parameters):
    from src.preprocess import preprocess_normal

    preprocess_normal(**load_ruamel(parameters))


@main.command("download")
@click.option("--parameters", help="Path to parameters file")
def download(parameters):
    from src.preprocess import download_artgraph

    download_artgraph(load_ruamel(parameters))


@main.command("extract_features")
@click.option("--parameters", help="Path to parameters file")
def extract_features(parameters):
    from src.features import extract_vis_clip
    extract_vis_clip(load_ruamel(parameters))

@main.command("test")
@click.option("--parameters", help="Path to parameters file")
def test(parameters):
    from src.experiment import test_clip

    test_clip(load_ruamel(parameters))


@main.command("experiment")
@click.option("--parameters", help="Path to parameters file")
def experiment(parameters):
    from src.experiment import fine_tune_clip

    fine_tune_clip(load_ruamel(parameters))
