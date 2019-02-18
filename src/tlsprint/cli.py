import os
import pkg_resources
import pickle

import click

from tlsprint.learn import learn_models
from tlsprint.identify import identify


@click.group()
def main():
    pass


@main.command("learn")
@click.argument("model_directory", type=click.Path(exists=True))
@click.argument("output", type=click.File("wb"))
def learn_command(model_directory, output):
    """Learn the model tree of all models in the specified directory and write
    the tree to 'output' as a pickled object."""
    tree = learn_models(model_directory)
    pickle.dump(tree, output)


@main.command("identify")
@click.argument("target")
@click.option("-p", "--target-port", default=443)
@click.option(
    "--model",
    help=(
        "Optional custom model to use (output from `learn`),"
        " defaults to model included in the distribution."
    ),
    type=click.File("rb"),
)
@click.option(
    "--graph-dir",
    help="Directory to store intermediate graphs, if desired.",
    type=click.Path(file_okay=False, writable=True),
)
def identify_command(target, target_port, model, graph_dir):
    """Uses the learned model to identify the implementation running on the
    target. By default this will use the model provided with the distribution,
    but a custom model can be supplied.
    """
    if model:
        tree = pickle.load(model)
    else:
        default_location = os.path.join("data", "model.p")
        tree = pickle.loads(pkg_resources.resource_string(__name__, default_location))

    tree.condense()
    groups = identify(tree, target, target_port, graph_dir)

    if groups:
        group = list(groups)[0]
        click.echo("Target has one of the following implementations:")
        click.echo("\n".join(sorted(tree.model_mapping[group])))
