import collections
import json
import operator
import pathlib


def _count_models_per_implementation(tls_path: pathlib.Path):
    model_paths = [p for p in tls_path.iterdir() if p.is_dir()]
    counts = collections.Counter()

    # For each model path, read the list of corresponding implementations and
    # version from the "versions.json"
    for path in model_paths:
        with open(path / "versions.json") as f:
            versions = json.load(f)

            # We are only interested in implementation names, not in version
            # numbers, so we use a set for this.
            implementations = set([name for name, _ in versions])

            # Add the names to the counter
            counts += collections.Counter(implementations)

    return counts


def summary(*, dedup_dir: str, **kwargs):
    """Return a summary of the number of unique models per implementation and
    TLS version as a dictionary. This uses the directory output from the
    deduplication step.
    """
    dedup_path = pathlib.Path(dedup_dir)
    tls_paths = [p for p in dedup_path.iterdir() if p.is_dir()]

    # The dedup directory is grouped by TLS version, but in this summary we
    # want to group by implementation. This means we have to invert this
    # grouping, and keeping track of the counts.
    counts = collections.defaultdict(dict)
    for path in sorted(tls_paths):
        tls_version = path.name

        # For each TLS version, extract the number of models per implementation
        for implementation, model_count in _count_models_per_implementation(
            path
        ).items():
            # For each implementation, add the counts to the corresponding TLS
            # version
            counts[implementation][tls_version] = model_count

    summary = [
        {"Name": implementation, **values} for implementation, values in counts.items()
    ]

    # Return the results, sorted by implementation name
    return sorted(summary, key=operator.itemgetter("Name"))
