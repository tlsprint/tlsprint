"""The learning component of tlsprint. The functions in this module can learn
from the output of StateLearning and create a model of all TLS implementations
in order to perform fingerprinting.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path


def _append_servers(tree, node, servers):
    try:
        tree.nodes[node]['servers']
    except KeyError:
        tree.nodes[node]['servers'] = set()
    tree.nodes[node]['servers'].update(servers)


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
    # Any node that has a itself as its only neighbor is a final node, this is
    # the base case for the recursion.
    if list(graph[current_node]) == [current_node]:
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
            _append_servers(tree, received_node, servers)

    # If not the base case, merge the current node into the tree and
    # recursively merge the rest
    else:
        for neighbor in graph[current_node]:
            for edge_number in graph[current_node][neighbor]:
                edge = graph[current_node][neighbor][edge_number]

                # Split the label in the sent and received message. Remove the
                # double quotes and the excess whitespace.
                sent, received = [
                    x.replace('"', '').strip() for x in edge['label'].split('/')  # noqa: E501
                ]

                # Append the sent and received messages to the tree
                sent_node = root + (sent, )
                received_node = root + (sent, received)
                tree.add_edge(root, sent_node, label=sent)
                tree.add_edge(sent_node, received_node, label=received)

                # If the received message contains 'ConnectionClosed', or
                # consists of a single dash ('-', which means time-out) this
                # path can be stopped here. This greatly reduces the number
                # of redundant nodes, because of 'ConnectionClosed' edges
                # go the the final node, which always contains many self loops.
                if 'ConnectionClosed' in received or '-' == received:
                    # Append the servers
                    _append_servers(tree, received_node, servers)

                    # Do not recurse
                    continue

                # It can happen that a model contains a loop. In this case,
                # append an additional node with the label 'LOOP' to the path,
                # add the servers to this node and do not recurse.
                if current_node == neighbor:
                    loop_node = received_node + ('LOOP', )
                    tree.add_edge(received_node, loop_node, label='LOOP')

                    # Append the servers
                    _append_servers(tree, loop_node, servers)

                    # Do not recurse
                    continue

                # Recurse with new root and current node
                tree = _merge_graph(tree, received_node, graph, neighbor, servers)  # noqa: E501

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
