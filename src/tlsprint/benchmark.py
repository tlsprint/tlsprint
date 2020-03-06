import collections
import copy
import pathlib

import pandas
import seaborn
from matplotlib import pyplot
from tlsprint.identify import INPUT_SELECTORS, identify
from tlsprint.trees import trees


def benchmark(tree, selector=INPUT_SELECTORS["first"]):
    """Return the inputs and outputs used to identify each model in the
    tree."""
    models = tree.models
    results = []
    for model in sorted(models):
        tree_copy = copy.deepcopy(tree)
        path = identify(tree_copy, model, benchmark=True, selector=selector)
        results.append(
            {
                "model": model,
                "implementations": sorted(tree.model_mapping[model]),
                "path": path,
            }
        )
    return results


def benchmark_all():
    benchmark_inputs = []
    for tree_type, tls_versions in trees.items():
        for version, tree in tls_versions.items():
            if tree_type == "adg":
                # The ADG tree type has no use for different selectors, as
                # there is always only one input possible.
                selectors = ("first",)
            else:
                selectors = INPUT_SELECTORS.keys()

            for selector in selectors:
                benchmark_inputs.append(
                    {
                        "type": tree_type,
                        "version": version,
                        "tree": tree,
                        "selector": selector,
                    }
                )

    results = []
    for info in benchmark_inputs:
        benchmark_result = benchmark(info["tree"], INPUT_SELECTORS[info["selector"]])
        results.append(
            {
                "type": info["type"],
                "version": info["version"],
                "selector": info["selector"],
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


def visualize_tls_group(benchmark_data, output_directory, title, weight_function):
    output_path = output_directory / f"{title}.pdf"
    pyplot.title(title)

    data = pandas.DataFrame()
    for entry in benchmark_data:
        model_values = [count_inputs(model) for model in entry["benchmark"]]
        model_weights = [weight_function(model) for model in entry["benchmark"]]
        for value, weight in zip(model_values, model_weights):
            if entry["type"].lower() == "adg":
                name = entry["type"].upper()
            else:
                name = f"{entry['type'].upper()} {entry['selector']}"
            for value, weight in zip(model_values, model_weights):
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

    # Group by TLS version
    grouped_by_tls = collections.defaultdict(list)
    for entry in benchmark_data:
        grouped_by_tls[entry["version"]].append(entry)

    for tls_version, entries in grouped_by_tls.items():
        for name, weight_function in (
            ("models", equal_model_weight),
            ("implementations", implementation_count_weight),
        ):
            visualize_tls_group(
                entries, output_directory, f"{tls_version} {name}", weight_function
            )
