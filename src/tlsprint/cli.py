import json
import pickle
import sys
from distutils.version import LooseVersion
from pathlib import Path

import click
import tabulate
from tlsprint import stats, util
from tlsprint.benchmark import benchmark_all
from tlsprint.identify import identify
from tlsprint.learn import (
    SUPPORTED_TREE_TYPES,
    _dot_to_networkx,
    construct_tree_from_dedup,
)


@click.group()
def main():
    pass


@main.command("learn")
@click.argument("dedup_directory", type=click.Path(exists=True))
@click.argument("output", type=click.File("wb"))
@click.option("--tree-type", default="adg", type=click.Choice(SUPPORTED_TREE_TYPES))
def learn_command(dedup_directory, output, tree_type):
    """Learn the distinguishing sequences based on the output of the `dedup`
    command. Write the resulting tree to 'output' as a pickled object."""
    tree = construct_tree_from_dedup(dedup_directory, tree_type=tree_type)
    pickle.dump(tree, output)


@main.command("identify")
@click.argument("target")
@click.option("-p", "--target-port", default=443)
@click.option(
    "--tree",
    help=(
        "Optional custom tree to use (output from `learn`),"
        " defaults to tree included in the distribution."
    ),
    type=click.File("rb"),
)
@click.option(
    "--graph-dir",
    help="Directory to store intermediate graphs, if desired.",
    type=click.Path(file_okay=False, writable=True),
)
def identify_command(target, target_port, tree, graph_dir):
    """Uses the learned tree to identify the implementation running on the
    target. By default this will use the tree provided with the distribution,
    but a custom tree can be supplied.
    """
    from tlsprint import trees

    if not tree:
        # For now, default to ADG TLS12
        tree = trees.trees["adg"]["TLS12"]
    else:
        tree = pickle.load(tree)

    tree.condense()
    models = identify(tree, target, target_port, graph_dir)

    if models:
        model = list(models)[0]
        version_info = tree.model_mapping[model]
        version_info = sorted(version_info, key=lambda x: LooseVersion(x[1]))
        version_strings = [" ".join(info) for info in version_info]
        click.echo("Target has one of the following implementations:")
        click.echo("\n".join(version_strings))
    else:
        click.echo("Failed to identify implementation")
        sys.exit(1)


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
    json.dump(converted, output_file, indent=4)


@main.command("dedup")
@click.argument("model_directory", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path())
def dedup_command(model_directory, output_directory):
    """Deduplicate the models directory.

    This reads the directory and assumes the path format
    `implementation/version/tls_version/learnedModel.dot` for every model. For
    every TLS protocol version, it groups together models which are the same.
    It then creates a directory for each TLS version, containing a directory
    for every unique model. Each model directory then contains the model in
    both Graphviz and JSON format, and a JSON file which lists the
    corresponding implementations and versions.
    """
    results = util.dedup_model_dir(model_directory)
    output_path = Path(output_directory)
    for protocol, models in results.items():
        for index, (model, versions) in enumerate(models.items()):
            model_name = f"model-{index + 1}"
            model_dir = output_path / protocol / model_name

            # Create the directory
            model_dir.mkdir(parents=True, exist_ok=True)

            # Write the model to this directory, both in Graphviz and JSON
            # format.
            with open(model_dir / "model.gv", "w") as f:
                f.write(model)

            graph = _dot_to_networkx(model)
            graph = util.prefix_nodes(graph, f"{model_name}_")
            converted = util.convert_graph(graph, add_resets=True)
            with open(model_dir / "model.json", "w") as f:
                json.dump(converted, f, indent=4)

            # Add the version list
            with open(model_dir / "versions.json", "w") as f:
                json.dump(versions, f, indent=4)


@main.command("draw")
@click.argument("graph", type=click.File("rb"))
@click.argument("output", type=click.File("wb"))
@click.option("--format", "fmt", default="svg")
def daw_command(graph, output, fmt):
    graph = pickle.load(graph)
    drawing = graph.draw(fmt=fmt)
    output.write(drawing)


@main.command("stats")
@click.option(
    "--type",
    "stats_type",
    multiple=True,
    type=click.Choice(stats.TYPES),
    help="Stats type to print.",
)
@click.option(
    "--model-directory",
    default="models/models",
    type=click.Path(exists=True),
    help="Directory where the models are stored.",
)
@click.option(
    "--dedup-directory",
    default="dedup",
    type=click.Path(exists=True),
    help="Directory where the deduplicated model are stored.",
)
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def stats_command(stats_type, model_directory, dedup_directory, fmt):
    """Provide statistics about the available models (number of
    implementations, unique models, etc.).
    """
    for type_ in stats_type:
        summary = stats.TYPE_HANDLERS[type_](
            models_dir=model_directory, dedup_dir=dedup_directory
        )

        if fmt == "table":
            click.echo(tabulate.tabulate(summary, headers="keys"))
        else:
            click.echo(json.dumps(summary))
        click.echo()


@main.command("benchmark")
def benchmark_command():
    results = benchmark_all()
    print(json.dumps(results, indent=4))
