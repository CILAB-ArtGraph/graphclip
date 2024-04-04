import click
from src.utils import load_ruamel


@click.group()
def main():
    pass

@main.command("split")
@click.option(
    "--parameters", help="Path to parameters file"
)
def split(parameters):
    from src.preprocess import preprocess_normal
    preprocess_normal(**load_ruamel(parameters))