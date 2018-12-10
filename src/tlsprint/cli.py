import click

from tlsprint.learn import learn_models


@click.group()
def main():
    pass


@main.command()
@click.argument('model_directory', type=click.Path(exists=True))
@click.argument('output')
def learn(model_directory, output):
    tree = learn_models(model_directory)
    tree.draw(output)
