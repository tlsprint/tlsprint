from .. import trees


def summary(**kwargs):
    summary = []
    for tree_type, tls_tree_dict in trees.trees.items():
        details = {"Type": tree_type}
        for tls_version, tree in sorted(tls_tree_dict.items()):
            details[tls_version] = len(tree)
        summary.append(details)
    return summary
