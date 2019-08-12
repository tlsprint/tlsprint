import json
import os
import pickle

import pkg_resources

import click
from tlsprint import util
from tlsprint.identify import identify
from tlsprint.learn import _dot_to_networkx, learn_models


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


@main.command("convert")
@click.argument("input_file", metavar="INPUT", type=click.File("r"))
@click.argument("output_file", metavar="OUTPUT", type=click.File("w"))
@click.option("--name", help="Prefix every node with this name")
@click.option(
    "--add-resets",
    help="Add a 'RESET / -' edge from every sinkhole to the start state.",
    is_flag=True,
)
def convert_command(input_file, output_file, name, add_resets):
    """Convert a graph from DOT to JSON.

    This is tailored to convert DOT output from LearnLib to the JSON files used
    by adg-finder. As such, it makes certain assumptions about the structure of
    the graph. For example, it assumes there is a dummy state called (often
    called `__start`), which is the only state without any incoming edges. This
    state is used to find the start state and will then be removed.
    """
    graph = _dot_to_networkx(input_file.read())

    # If a name is specified, prefix all nodes with that name
    if name:
        graph = util.prefix_nodes(graph, f"{name}_")

    converted = util.convert_graph(graph, add_resets=add_resets)
    print(json.dumps(converted, indent=4), file=output_file)
