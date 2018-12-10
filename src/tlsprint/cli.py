import click

from tlsprint.learn import learn_models
from tlsprint.identify import identify


@click.group()
def main():
    pass


@main.command('learn')
@click.argument('model_directory', type=click.Path(exists=True))
@click.argument('output')
def learn_command(model_directory, output):
    """Learn the model tree of all models in the specified directory. Returns
    a DOT file with the model tree."""
    tree = learn_models(model_directory)
    tree.draw(output)


@main.command('identify')
@click.argument('model_directory', type=click.Path(exists=True))
def identify_command(model_directory):
    """Learn the model and identify the implementation. Assumes
    TLSAttackerConnector is running on port 6666 of the localhost, pointing the
    the target to identify."""
    tree = learn_models(model_directory)
    tree.condense()
    groups = identify(tree)

    if groups:
        click.echo('Target belongs to one of the following groups:')
        for i, group in enumerate(groups):
            click.echo(f'Group {i + 1}:')
            click.echo('\n'.join(tree.model_mapping[group]))
            click.echo()
