import collections
import copy
import itertools
import pathlib

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from tlsprint import util
from tlsprint.identify import INPUT_SELECTORS, MODEL_WEIGHTS, identify
from tlsprint.trees import trees


def count_inputs(messages):
    return len(messages) // 2


def count_resets(messages):
    return messages.count("RESET")


PATH_VALUES = {
    "inputs": count_inputs,
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


def visualize(benchmark_data, output_directory, version, weight_function):
    version_string = util.format_tls_string(version)
    file_name = f"{version} {weight_function}.pdf"
    output_path = output_directory / file_name

    data = pandas.DataFrame()
    for entry in benchmark_data:
        name = f"{entry['type'].upper()}"
        if entry["type"].lower() != "adg":
            name += f" {entry['selector']}"

        for item in entry["benchmark"]:
            for metric, value in item["values"].items():
                data = data.append(
                    [
                        {"name": name, "value": value, "metric": metric,}
                        for _ in range(item["weight"])
                    ],
                    ignore_index=True,
                )

    seaborn.violinplot(
        x="name",
        y="value",
        data=data,
        bw=0.1,
        scale="count",
        hue="metric",
        split=True,
        inner=None,
    )
    seaborn.pointplot(
        x="name",
        y="value",
        data=data,
        estimator=numpy.mean,
        join=False,
        hue="metric",
        palette="bright",
        capsize=0.1,
        legend=False,
    )

    title = f"{version_string} - Model weight: {weight_function.capitalize()}"
    pyplot.title(title)
    pyplot.xlabel("Identification method")
    pyplot.ylabel("Metric value")
    pyplot.savefig(output_path)
    pyplot.clf()


def visualize_tls_group(benchmark_data, output_directory, version):
    for weight_function in MODEL_WEIGHTS.keys():
        subset = [
            entry for entry in benchmark_data if entry["weight"] == weight_function
        ]
        visualize(subset, output_directory, version, weight_function)

    return


def visualize_all(benchmark_data, output_directory):
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(exist_ok=True)
    seaborn.set(style="dark", palette="pastel", color_codes=True)

    tls_versions = {entry["version"] for entry in benchmark_data}
    for version in sorted(tls_versions):
        subset = [entry for entry in benchmark_data if entry["version"] == version]
        visualize_tls_group(subset, output_directory, version)
    return
