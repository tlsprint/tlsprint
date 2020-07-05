import pathlib

from .. import learn


def _tls_summary(tls_path: pathlib.Path):
    summary = {"TLS version": tls_path.name}

    # Count the number of unique models for this TLS version
    model_paths = [p for p in tls_path.iterdir() if p.is_dir()]
    summary["Unique models"] = len(model_paths)

    # Read all the models and add their size
    model_sizes = []
    for path in model_paths:
        with open(path / "model.gv") as f:
            graph = learn._dot_to_networkx(f.read())

            # Compensate for the dummy __start node
            model_sizes.append(len(graph) - 1)

    # Add some statistics about the models sizes
    summary["Average model size"] = round(sum(model_sizes) / len(model_sizes), 1)
    summary["Largest model"] = max(model_sizes)
    summary["Smallest model"] = min(model_sizes)

    return summary


def summary(*, dedup_dir: str, **kwargs):
    """Return a summary of the number of unique models per TLS version and as
    a dictionary, it also includes some information about the size of the
    models. This uses the directory output from the deduplication step.
    """
    dedup_path = pathlib.Path(dedup_dir)
    tls_paths = [p for p in dedup_path.iterdir() if p.is_dir()]

    summary = []
    for path in sorted(tls_paths):
        summary.append(_tls_summary(path))

    return summary
