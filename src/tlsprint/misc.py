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
    -   Set the label of all leaf nodes to the list of servers, including
        the percentage of how much servers are contained in this leaf,
        compared to all present in the tree.

    Args:
        tree: The tree to modify and draw.
        path: The path where to store the DOT file.
    """
    # Compute the total number of servers included in this tree
    all_servers = set()
    for node in tree.nodes:
        if tree.out_degree(node) == 0:
            all_servers.update(tree.nodes[node]['servers'])
    server_count = len(all_servers)

    # Relabel all the nodes
    for node in tree.nodes:
        if tree.out_degree(node) == 0:
            # Leaf node
            servers = sorted(tree.nodes[node]['servers'])
            server_share = '{:.2f}%'.format(100 * len(servers) / server_count)
            tree.nodes[node]['label'] = '\n'.join([server_share] + servers)
            tree.nodes[node]['shape'] = 'rectangle'
        else:
            # Not a leaf node
            tree.nodes[node]['label'] = ''
    networkx.drawing.nx_pydot.write_dot(tree, path)
