"""The learning component of tlsprint. The functions in this module can learn
from the output of StateLearning and create a model of all TLS implementations
in order to perform fingerprinting.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path

import networkx


def _merge_graph(tree, root, graph, current_node, servers):
    """Merge a single model graph into the tree, used by `learn_models` to
    create the tree of all merged models.

    Args:
        tree: The tree that is being constructed, and into which the graph will
            be merged.
        root: The current root node of the tree, the graph will be merged
            beginning at this node.
        graph: The graph to merge into the tree
        current_node: The current node of the graph to merge
        servers: The server implementations corresponding to this graph.
    """
    # Any node that has a self loop is a final node, this is the base case
    # for the recursion.
    if current_node in networkx.nodes_with_selfloops(graph):
        for edge_number in graph[current_node][current_node]:
            edge = graph[current_node][current_node][edge_number]

            # Split the label in the sent and received message. Remove the
            # double quotes and the excess whitespace.
            sent, received = [
                x.replace('"', '').strip() for x in edge['label'].split('/')
            ]

            # Append the sent and received messages to the tree
            sent_node = root + (sent, )
            received_node = root + (sent, received)
            tree.add_edge(root, sent_node, label=sent)
            tree.add_edge(sent_node, received_node, label=received)

            # Append the servers to the final node as attribute
            try:
                tree.nodes[received_node]['servers']
            except KeyError:
                tree.nodes[received_node]['servers'] = set()
            tree.nodes[received_node]['servers'].update(servers)
    else:
        pass

    return tree


def learn_models(directory):
    """Learn the complete tree from all the different models.

    Args:
        directory: A directory that contains the output of StateLearner. It
            expects the directory to contain directories with the names of the
            different TLS implementations, where each TLS directory contains a
            file 'learnedModel.dot'. Will skip implementations where this file
            is absent.

    Returns:
        model_tree: A networkx tree containing the paths for all models.
    """
    logger = logging.getLogger()

    # Read and deduplicate the models
    server_dirs = [f.name for f in os.scandir(directory) if f.is_dir()]
    models = defaultdict(list)

    for server in server_dirs:
        try:
            model_root = Path(directory)
            with (model_root / server / 'learnedModel.dot').open() as f:
                models[f.read()].append(server)

                logger.info(f'Found model for: {server}')
        except OSError:
            logger.warning(f'Could not find model for: {server}')

    return models
