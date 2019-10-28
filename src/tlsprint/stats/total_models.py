import collections
import pathlib


def _implementation_summary(implementation_path: pathlib.Path):
    summary = {"Name": implementation_path.name}

    version_paths = [p for p in implementation_path.iterdir() if p.is_dir()]

    # To count the number of models for a given implementation, we need a list
    # of the supported TLS versions for each implementation version.
    tls_versions = [x.name for p in version_paths for x in p.iterdir()]
    tls_counts = collections.Counter(tls_versions)

    # Merge the counts with the existing summary info
    summary = {**summary, **tls_counts}

    # Add a total count column
    summary["Total"] = sum(tls_counts.values())

    return summary


def summary(*, models_dir: str = None, **kwargs):
    """Return a summary of the number of learned models per implementation and
    TLS version as a dictionary. This uses the directory containing the raw
    models.
    """
    models_path = pathlib.Path(models_dir)
    implementation_paths = [p for p in models_path.iterdir() if p.is_dir()]

    summary = []
    for path in sorted(implementation_paths):
        summary.append(_implementation_summary(path))

    return summary
