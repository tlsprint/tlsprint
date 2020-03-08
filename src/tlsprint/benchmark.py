import collections
import copy
import itertools
import pathlib

import pandas
import seaborn
from matplotlib import pyplot
from tlsprint.identify import INPUT_SELECTORS, MODEL_WEIGHTS, identify
from tlsprint.trees import trees


def count_inputs(messages):
    return len(messages) // 2


def count_resets(messages):
    return messages.count("RESET")


PATH_VALUES = {
    "count": count_inputs,
    "resets": count_resets,
}


def benchmark_model(tree, model, selector, weight_function):
    tree_copy = copy.deepcopy(tree)
    path = identify(
        tree_copy,
        model,
        benchmark=True,
        selector=selector,
        weight_function=weight_function,
    )
    return {name: value(path) for name, value in PATH_VALUES.items()}


def benchmark(tree, selector, weight_function):
    """Return the inputs and outputs used to identify each model in the
    tree."""
    models = tree.models
    if selector == INPUT_SELECTORS["random"]:
        iterations = 20
    else:
        iterations = 1

    results = []
    for model in sorted(models):
        path_values = []
        for _ in range(iterations):
            path_values.append(benchmark_model(tree, model, selector, weight_function))

        # Compute averages of path values
        values_sums = collections.defaultdict(int)
        for values in path_values:
            for name, value in values.items():
                values_sums[name] += value
        averages = {name: sum / len(path_values) for name, sum in values_sums.items()}
        results.append(
            {
                "model": model,
                "weight": weight_function(tree.model_mapping[model]),
                "values": averages,
            }
        )
    return results


def benchmark_all():
    benchmark_inputs = []
    for tree_type, tls_versions in trees.items():
        for version, tree in tls_versions.items():
            selectors = INPUT_SELECTORS.keys()
            weight_functions = MODEL_WEIGHTS.keys()

            if tree_type == "adg":
                # The ADG tree type has no use for different selectors or
                # weight functions, as there is always only one input
                # possible.
                selectors = ("first",)

            for selector, weight in itertools.product(selectors, weight_functions):
                benchmark_inputs.append(
                    {
                        "type": tree_type,
                        "version": version,
                        "tree": tree,
                        "selector": selector,
                        "weight": weight,
                    }
                )

    results = []
    for info in benchmark_inputs:
        benchmark_result = benchmark(
            info["tree"],
            INPUT_SELECTORS[info["selector"]],
            MODEL_WEIGHTS[info["weight"]],
        )
        results.append(
            {
                "type": info["type"],
                "version": info["version"],
                "selector": info["selector"],
                "weight": info["weight"],
                "benchmark": benchmark_result,
            }
        )

    return results


def count_inputs(model_info):
    return len(model_info["path"]) // 2


def equal_model_weight(model_info):
    return 1


def implementation_count_weight(model_info):
    return len(model_info["implementations"])


def visualize_tls_group(benchmark_data, output_directory, version):
    print(len(benchmark_data))
    return
    output_path = output_directory / f"{title}.pdf"
    pyplot.title(title)

    data = pandas.DataFrame()
    for entry in benchmark_data:

        name = f"{entry['type'].upper()} {entry['weight']}"
        if entry["type"].lower() != "adg":
            name += f" {entry['selector']}"

        for item in entry["benchmark"]:
            value = count_inputs(item)
            weight = item["weight"]
            data = data.append(
                [{"name": name, "value": value,} for _ in range(weight)],
                ignore_index=True,
            )

    output_path = output_directory / f"{title}.pdf"
    seaborn.violinplot(x="name", y="value", data=data, showmeans=True)
    pyplot.savefig(output_path)
    pyplot.clf()


def visualize_all(benchmark_data, output_directory):
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(exist_ok=True)

    tls_versions = {entry["version"] for entry in benchmark_data}
    for version in sorted(tls_versions):
        subset = [entry for entry in benchmark_data if entry["version"] == version]
        visualize_tls_group(subset, output_directory, version)
    return

    # Group by TLS version
    grouped_by_tls = collections.defaultdict(list)
    for entry in benchmark_data:
        grouped_by_tls[entry["version"]].append(entry)

    for tls_version, entries in grouped_by_tls.items():
        # Group by TLS version
        grouped_by_weight_function = collections.defaultdict(list)
        for entry in benchmark_data:
            grouped_by_weight_function[entry["weight"]].append(entry)

        for weight_function, sub_entries in grouped_by_weight_function.items():

            visualize_tls_group(
                sub_entries, output_directory, f"{tls_version} {weight_function}"
            )
