"""This module queries the available trees and returns them trough the `trees`
object.
"""

import pickle
from pathlib import Path

import pkg_resources


def _read_trees():
    data_path = Path(pkg_resources.resource_filename(__name__, "data"))
    type_paths = [p for p in data_path.iterdir() if p.is_dir()]

    trees = {}
    for type_path in type_paths:
        trees[type_path.name] = {}
        for path in type_path.iterdir():
            tree_name = path.name.split(".")[0]
            with open(path, "rb") as f:
                trees[type_path.name][tree_name] = pickle.load(f)

    return trees


trees = _read_trees()
