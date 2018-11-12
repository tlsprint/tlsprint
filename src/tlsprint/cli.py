import click

from tlsprint.learn import learn_models
from tlsprint.misc import draw_model_tree


@click.group()
def main():
    pass


@main.command()
@click.argument('model_directory', type=click.Path(exists=True))
@click.argument('output')
def learn(model_directory, output):
    tree = learn_models(model_directory)
    draw_model_tree(tree, output)
