"""Miscellaneous function that are used by tlprint, but not (yet)
categorized in one of the other modules.
"""

import networkx


def draw_model_tree(tree, path):
    """Draw the tree created by the `learn_model` function by outputting a
    Graphviz file in DOT format. This slightly modifies the tree in order to
    improve the output:
    -   Set the label of all non leafs nodes to blank, as the information is
        already captured by the edges.
    -   Set the label of all leaf nodes to the list of servers.

    Args:
        tree: The tree to modify and draw.
        path: The path where to store the DOT file.
    """
    for node in tree.nodes:
        if tree.out_degree(node) == 0:
            # Leaf node
            servers = sorted(tree.nodes[node]['servers'])
            tree.nodes[node]['label'] = '\n'.join(servers)
        else:
            # Not a leaf node
            tree.nodes[node]['label'] = ''
    networkx.drawing.nx_pydot.write_dot(tree, path)
