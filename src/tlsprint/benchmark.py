import collections
import copy
import itertools
import multiprocessing
import operator
import pathlib
import time

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


def benchmark(info):
    """Return the inputs and outputs used to identify each model in the
    tree."""
    tree = info["Tree"]
    model = info["Model"]
    selector = INPUT_SELECTORS[info["Input selector"]]
    weight_function = MODEL_WEIGHTS[info["Weight function"]]
    iterations = info["Iterations"]

    results = []
    for _ in range(iterations):
        results.append(benchmark_model(tree, model, selector, weight_function))

    # Copy most, but not all values from the info object
    benchmark_results = {
        key: value for key, value in info.items() if key not in ("Tree", "Iterations")
    }

    # Add the results and return it
    benchmark_results["Results"] = results

    return benchmark_results


def generate_benchmark_inputs(iterations):
    """Generate benchmarks as a Pandas DataFrame, where each row is lists the
    input of one benchmark test."""
    # We initialize a Pandas DataFrame with a list of all the trees, the
    # resulting DataFrame has the following columns:
    # - Tree type: This is the tree type, ADG or HDT
    # - TLS version: The TLS version corresponding to the tree.
    # - Tree: A reference to the tree itself.
    # - Model: A list of models names for this tree.
    tree_list = []
    for tree_type, tls_versions in trees.items():
        for version, tree in tls_versions.items():
            tree_list.append(
                {
                    "Tree type": tree_type,
                    "TLS version": version,
                    "Tree": tree,
                    "Model": list(tree.models),
                }
            )
    df = pandas.DataFrame(tree_list)

    # Instead of a list of models, we want a separate row for each model with
    # the rest of the values the same. This can be done with the `explode`
    # function.
    df = df.explode("Model")

    # We now have the basic dataset with all the trees and models that we want
    # to benchmark. The next step we perform, is adding input selectors. For
    # this we create a mapping between selectors and methods, which defaults to
    # using all input selectors. For ADG we only add "first", because there is
    # only one path and more input selection is irrelevant.
    input_selector_mapping = collections.defaultdict(
        lambda: list(INPUT_SELECTORS.keys())
    )
    input_selector_mapping["adg"] = ["first"]

    # We then create a dataframe from this mapping.
    input_selector_df = pandas.DataFrame(
        [
            {"Tree type": method, "Input selector": input_selector_mapping[method]}
            for method in df["Tree type"].unique()
        ]
    )
    # We merge this mapping and explode the Input selector column to get the
    # extended benchmarks.
    df = df.merge(input_selector_df, on="Tree type", how="left")
    df = df.explode("Input selector")

    # To distinguish between the HDT with different input selectors, we add the
    # "Method" field, which is the name of the Tree type in upper case with the
    # input selector. For ADG we only add the tree type
    not_adg_rows = df["Tree type"] != "adg"
    df["Method"] = "ADG"
    df["Method"][not_adg_rows] = (
        df["Tree type"][not_adg_rows].apply(str.upper)
        + " "
        + df["Input selector"][not_adg_rows].apply(str.capitalize)
    )

    # Some input selectors can use multiple weight functions. For now this only
    # applies to the Gini input selector, the rest is unaffected by weight and
    # is assigned the most simple "equal" function. We still evaluate for
    # different weights functions during the analysis, but running the
    # benchmark for different weight functions for "First" and "Random" is
    # merely a duplication.
    #
    # We apply the same trick as above to map the input selectors to the weight
    # functions. The default here is to only use the "equal" weight function.
    weight_function_mapping = collections.defaultdict(lambda: ["equal"])
    weight_function_mapping["gini"] = list(MODEL_WEIGHTS.keys())

    weight_function_df = pandas.DataFrame(
        [
            {
                "Input selector": selector,
                "Weight function": weight_function_mapping[selector],
            }
            for selector in df["Input selector"].unique()
        ]
    )

    # Merge the mapping and explode
    df = df.merge(weight_function_df, on="Input selector", how="left")
    df = df.explode("Weight function")

    # To distinguish between the different combinations of Gini and weight
    # functions, we add the weight function to the Method field.
    gini_rows = df["Input selector"] == "gini"
    df["Method"][gini_rows] = (
        df["Method"][gini_rows] + " (" + df["Weight function"][gini_rows] + ")"
    )

    # Lastly, to pass the iteration count to the benchmark function, add
    # a column to the dataframe.
    df["Iterations"] = iterations

    # Convert the dataframe to a list, as this is easier to distribute over
    # multiple processes.

    return df.to_dict(orient="records")


def benchmark_all(iterations=100):
    benchmark_inputs = generate_benchmark_inputs(iterations)

    # Apply the benchmark function to the inputs and show a progress bar.
    input_count = len(benchmark_inputs)
    with multiprocessing.Pool() as p:
        results = list(
            tqdm.tqdm(p.imap(benchmark, benchmark_inputs), total=input_count)
        )

    return results


def _group_data(data, fields):
    """Sort and group the data by the specified fields"""
    key_function = operator.itemgetter(*fields)
    data = sorted(data, key=key_function)
    grouped_data = {
        key: list(values) for key, values in itertools.groupby(data, key=key_function)
    }
    return grouped_data


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
    seaborn.displot(x=metric["name"], col=label["name"], data=df)
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
