import copy

from tlsprint.identify import INPUT_SELECTORS, identify
from tlsprint.trees import trees


def benchmark(tree, selector=INPUT_SELECTORS["first"]):
    """Return the inputs and outputs used to identify each model in the
    tree."""
    models = tree.models
    results = []
    for model in sorted(models):
        tree_copy = copy.deepcopy(tree)
        path = identify(
            tree_copy, model, benchmark=True, selector=INPUT_SELECTORS["first"]
        )
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
