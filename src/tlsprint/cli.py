import click

from tlsprint.learn import learn_models


@click.group()
def main():
    pass


@main.command()
@click.argument('model_directory', type=click.Path(exists=True))
def learn(model_directory):
    click.echo('\n'.join(learn_models(model_directory)))
