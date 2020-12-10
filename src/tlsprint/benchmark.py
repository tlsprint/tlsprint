import collections
import copy
import itertools
import multiprocessing
import operator
import pathlib
import time

import numpy
import pandas
import seaborn
import tqdm
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
    candidate_models, path = identify(
        tree_copy,
        model,
        benchmark=True,
        selector=selector,
        weight_function=weight_function,
    )
    end_time = time.perf_counter()

    # If the desired model is not in the list of leaf model,
    # something went wrong and we should abort.
    if model not in candidate_models:
        raise RuntimeError("Benchmark model does not match found model!")

    results = {name: value(path) for name, value in PATH_VALUES.items()}
    results["time"] = end_time - start_time

    return results


def benchmark(info, iterations=20):
    """Return the inputs and outputs used to identify each model in the
    tree."""
    tree = info["tree"]
    model = info["model"]
    selector = INPUT_SELECTORS[info["selector"]]
    weight_function = MODEL_WEIGHTS[info["weight"]]

    path_values = []
    for _ in range(iterations):
        path_values.append(benchmark_model(tree, model, selector, weight_function))

    # Compute averages of path values
    values_sums = collections.defaultdict(int)
    for values in path_values:
        for name, value in values.items():
            values_sums[name] += value
    averages = {name: sum / len(path_values) for name, sum in values_sums.items()}

    return {
        "tree_type": info["tree_type"],
        "tls_version": info["tls_version"],
        "model": info["model"],
        "selector": info["selector"],
        "weight": info["weight"],
        "result": {
            "weight": weight_function(tree.model_mapping[model]),
            "values": averages,
        },
    }


def generate_benchmark_inputs():
    # First create a list of the available trees, one for each combination of
    # tree type and TLS version.
    trees_list = []
    for tree_type, tls_versions in trees.items():
        for version, tree in tls_versions.items():
            trees_list.append(
                {"tree_type": tree_type, "tls_version": version, "tree": tree}
            )

    # We then initialize the benchmark queue, containing every model from
    # every tree
    queue = []
    for values in trees_list:
        queue += [{**values, "model": model} for model in values["tree"].models]

    # Next, take the product of the queue with the input selectors.
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

    input_count = len(benchmark_inputs)
    with multiprocessing.Pool() as p:
        results = list(
            tqdm.tqdm(p.imap(benchmark, benchmark_inputs), total=input_count)
        )

    return results


def count_inputs(model_info):
    return len(model_info["path"]) // 2


def equal_model_weight(model_info):
    return 1


def implementation_count_weight(model_info):
    return len(model_info["implementations"])


def visualize(benchmark_data, output_directory, tls_version, weight_function):
    tls_version_string = util.format_tls_string(tls_version)
    file_name = f"{tls_version} {weight_function}.pdf"
    output_path = output_directory / file_name

    data = pandas.DataFrame()
    for entry in benchmark_data:
        name = f"{entry['tree_type'].upper()}"
        if name != "ADG":
            name += f" {entry['selector']}"

        for item in entry["result"].items():
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

    title = f"{tls_version_string} - Model weight: {weight_function.capitalize()}"
    pyplot.title(title)
    pyplot.xlabel("Identification method")
    pyplot.ylabel("Metric value")
    pyplot.savefig(output_path)
    pyplot.clf()


def _group_data(data, fields):
    """Sort and group the data by the specified fields"""
    key_function = operator.itemgetter(*fields)
    data = sorted(data, key=key_function)
    grouped_data = {
        key: list(values) for key, values in itertools.groupby(data, key=key_function)
    }
    return grouped_data


def visualize_tls_group(benchmark_data, output_directory, version):
    weights = {entry["weight"] for entry in benchmark_data}
    for weight_function in weights:
        subset = [
            entry for entry in benchmark_data if entry["weight"] == weight_function
        ]
        visualize(subset, output_directory, version, weight_function)

    return


def visualize_tls_version_weight(data, title, output_path):
    breakpoint()
    print(title, len(data))
    pass


def visualize_tls_version(tls_version, data, output_directory):
    # We want to compare the data for each weight function
    weight_key = operator.itemgetter("weight")
    data = sorted(data, key=weight_key)
    subsets = {
        weight_function: list(subdata)
        for weight_function, subdata in itertools.groupby(data, key=weight_key)
    }

    for weight_function, subdata in subsets.items():
        visualize_tls_version_weight(
            tls_version, weight_function, subdata, output_directory
        )


def visualize_subset(data, label, metric, title, output_path):
    # First we create a Pandas dataframe with the data to visualize. This
    # will include the label field, and the metric field from the result
    # values. To simulate the model weight, we add this item `weight`
    # times.
    df = pandas.DataFrame(columns=["label", "metric"])
    for entry in data:
        row = {
            label["name"]: entry[label["field"]],
            metric["name"]: entry["result"]["values"][metric["field"]],
        }
        values = [row for _ in range(entry["result"]["weight"])]
        df = df.append(values, ignore_index=True)

    # Sort the data to set the column order
    df = df.sort_values([label["name"]])

    # Create a plot of the data and save this figure.
    seaborn.set_theme(font_scale=2, style="whitegrid")
    pyplot.savefig(output_path.with_suffix(".pdf"))
    pyplot.close()

    # Create a markdown file with a table describing the same data
    grouped_df = df.groupby(label["name"])
    markdown = grouped_df[metric["name"]].describe().round(4).to_markdown()

    # Add caption to table
    markdown += "\n\n"
    markdown += f": Benchmark summary for {title}"
    with open(output_path.with_suffix(".md"), "w") as f:
        f.write(markdown)


def visualize_all(benchmark_data, output_directory):
    # Make sure the output directory exists
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(exist_ok=True)

    # We want to compare the performance of the different methods: the ADG,
    # and the HDT with different input selectors. We add a `method` field
    # which combines this information, which we will use in the plots
    # later.
    for entry in benchmark_data:
        entry["method"] = entry["tree_type"].upper()

        if entry["method"] != "ADG":
            entry["method"] += " " + entry["selector"].capitalize()

    # We want to compare the performance of the different methods for
    # across TLS version and for different weight functions. So we group
    # the data by the `tls_version` and `weight` fields.
    subsets = _group_data(benchmark_data, ["tls_version", "weight"])

    for key, values in subsets.items():
        for metric in [
            {"name": "Number of inputs", "field": "inputs"},
            {"name": "Number of resets", "field": "resets"},
            {"name": "Time in seconds", "field": "time"},
        ]:
            filename = " ".join(key) + f" {metric['field']}"
            output_path = output_directory / filename

            title = (
                f"{util.format_tls_string(key[0])}"
                f", model weight: {key[1]}"
                f", metric: {metric['field']}"
            )

            label = {"name": "Method", "field": "method"}

            visualize_subset(values, label, metric, title, output_path)
            # visualize_tls_version_weight(values, title, output_path)
    return

    # Data for different TLS versions is not comparable, so we group by
    # that first, for which the data needs to be sorted.
    tls_key = operator.itemgetter("tls_version")
    benchmark_data = sorted(benchmark_data, key=tls_key)
    subsets = {
        tls_version: list(subdata)
        for tls_version, subdata in itertools.groupby(benchmark_data, key=tls_key)
    }

    for tls_version, subdata in subsets.items():
        visualize_tls_version(tls_version, subdata, output_directory)

    return

    seaborn.set(style="dark", palette="pastel", color_codes=True)

    tls_versions = {entry["tls_version"] for entry in benchmark_data}
    for version in sorted(tls_versions):
        subset = [entry for entry in benchmark_data if entry["tls_version"] == version]
        visualize_tls_group(subset, output_directory, version)
    return
