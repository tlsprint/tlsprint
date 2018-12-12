import pickle

import click

from tlsprint.learn import learn_models
from tlsprint.identify import identify


@click.group()
def main():
    pass


@main.command('learn')
@click.argument('model_directory', type=click.Path(exists=True))
@click.argument('output', type=click.File('wb'))
def learn_command(model_directory, output):
    """Learn the model tree of all models in the specified directory and write
    the tree to 'output' as a pickled object."""
    tree = learn_models(model_directory)
    pickle.dump(tree, output)


@main.command('identify')
@click.argument('model', type=click.File('rb'))
def identify_command(model):
    """Uses the learned model to identify the implementation. Assumes
    TLSAttackerConnector is running on port 6666 of the localhost, pointing the
    the target to identify."""
    tree = pickle.load(model)
    tree.condense()
    groups = identify(tree)

    if groups:
        click.echo('Target belongs to one of the following groups:')
        for i, group in enumerate(groups):
            click.echo(f'Group {i + 1}:')
            click.echo('\n'.join(sorted(tree.model_mapping[group])))
            click.echo()
