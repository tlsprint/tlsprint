import copy
import pathlib

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


def _plot_histogram(data, tree_type, tls_version, selector, output_path, bins=None):
    mean = sum(data) / len(data)
    pyplot.hist(data, bins=bins)
    pyplot.axvline(mean, color="red")
    pyplot.savefig(output_path)
    pyplot.clf()


def visualize(tree_type, tls_version, selector, benchmark, output_directory, bins=None):
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(exist_ok=True)
    base_file_name = f"{tree_type}-{tls_version}-{selector}"

    # Generate a histogram of the number of input messages for each model
    input_counts = [len(model["path"]) // 2 for model in benchmark]

    output_path = output_directory / (base_file_name + "-models.pdf")
    _plot_histogram(input_counts, tree_type, tls_version, selector, output_path, bins)

    # Generate a histogram of the number of input messages for each
    # implementation
    implementation_counts = [len(model["implementations"]) for model in benchmark]
    inputs_per_implementation = []
    for inputs, implementations in zip(input_counts, implementation_counts):
        inputs_per_implementation += [inputs] * implementations

    output_path = output_directory / (base_file_name + "-implementations.pdf")
    _plot_histogram(
        inputs_per_implementation, tree_type, tls_version, selector, output_path, bins
    )


def visualize_all(benchmark_data, output_directory):
    # Determine the bin range for to histogram, for more easily comparing the
    # different plots.
    input_counts = []
    for entry in benchmark_data:
        input_counts += [len(model["path"]) // 2 for model in entry["benchmark"]]
    bins = range(min(input_counts), max(input_counts) + 1)

    for entry in benchmark_data:
        visualize(
            entry["type"],
            entry["version"],
            entry["selector"],
            entry["benchmark"],
            output_directory,
            bins=bins,
        )
