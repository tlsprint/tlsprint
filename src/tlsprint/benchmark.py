import collections
import copy
import itertools
import pathlib
import time

import numpy
import pandas
import seaborn
from matplotlib import pyplot

from . import util
from .identify import INPUT_SELECTORS
from .identify import MODEL_WEIGHTS
from .identify import identify
from .trees import trees


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

    start_time = time.perf_counter()
    path = identify(
        tree_copy,
        model,
        benchmark=True,
        selector=selector,
        weight_function=weight_function,
    )
    end_time = time.perf_counter()

    results = {name: value(path) for name, value in PATH_VALUES.items()}
    results["time"] = start_time - end_time

    return results


def benchmark(tree, selector, weight_function, info):
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


def generate_benchmark_inputs():
    # First create a list of the available trees, one for each combination of
    # tree type and TLS version.
    queue = []
    for tree_type, tls_versions in trees.items():
        for version, tree in tls_versions.items():
            queue.append({"tree_type": tree_type, "tls_version": version, "tree": tree})

    # Next, combine the queue with the input selectors.
    queue = [
        {**values, "selector": selector}
        for values, selector in itertools.product(queue, INPUT_SELECTORS.keys())
    ]

    # The ADG method doesn't benefit from input selectors as only one input
    # is available at each time. The "first" selector is therefore enough for
    # the ADG and we remove the others.
    def should_keep(values):
        if values["tree_type"] == "adg" and values["selector"] != "first":
            return False
        return True

    queue = [values for values in queue if should_keep(values)]

    # We then take the product of the queue with the weight functions
    queue = [
        {**values, "weight": weight}
        for values, weight in itertools.product(queue, MODEL_WEIGHTS.keys())
    ]

    return queue


def benchmark_all():
    benchmark_inputs = generate_benchmark_inputs()
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
                        {"name": name, "value": value, "metric": metric}
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
