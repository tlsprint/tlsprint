import collections
import json
import pathlib


def _model_path_number(path):
    return int(path.name.split("-")[-1])


def _model_summary(path):
    # Read the graph data
    with open(path / "model.json") as file:
        graph_data = json.load(file)

    # Read version data
    with open(path / "versions.json") as file:
        version_data = json.load(file)

    version_counts = collections.Counter(x[0] for x in version_data)
    return {
        "Model": path.name,
        "Number of states": len(graph_data["states"]),
        "OpenSSL versions": version_counts.get("openssl", 0),
        "mbed TLS versions": version_counts.get("mbedtls", 0),
    }


def summary(*, dedup_dir, tls_version, **kwargs):
    dedup_path = pathlib.Path(dedup_dir)
    tls_path = dedup_path / tls_version
    model_paths = [x for x in tls_path.iterdir() if x.is_dir()]

    # Sort paths by model number
    model_paths = sorted(model_paths, key=_model_path_number)

    summary = [_model_summary(path) for path in model_paths]

    return summary
