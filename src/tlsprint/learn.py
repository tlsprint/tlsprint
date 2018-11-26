"""The learning component of tlsprint. The functions in this module can learn
from the output of StateLearning and create a model of all TLS implementations
in order to perform fingerprinting.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path

import networkx
from networkx.algorithms import simple_cycles
import pydot


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

                # If the received message contains 'ConnectionClosed', this
                # path can be stopped here. This greatly reduces the number
                # of redundant nodes, because of 'ConnectionClosed' edges
                # go the the final node, which always contains many self loops.
                if 'ConnectionClosed' in received:
                    # Append the servers
                    _append_servers(tree, received_node, servers)

                    # Do not recurse
                    continue

                # It can happen that a model contains a cycle, this is detected
                # by checking if the current node is the final node in any of
                # the values returned by `simple_cycles`. When a cycle is
                # detected, append an additional node with the message that
                # causes the cycle, and include 'CYCLE' in the edge label, and
                # stop recursing.
                if current_node in [cycle[-1] for cycle in simple_cycles(graph)]:  # noqa: E501
                    loop_node = received_node + ('CYCLE', )
                    tree.add_edge(received_node, loop_node, label='CYCLE')

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
        tree: A networkx tree containing the paths for all models.
    """
    logger = logging.getLogger()

    # Collect the names of the server directories
    server_dirs = [f.name for f in os.scandir(directory) if f.is_dir()]

    # Merge duplicates toghether, by using the DOT model as dictionary key,
    # and the server list as the value.
    models = defaultdict(list)

    for server in server_dirs:
        # Not all directories actually contain a model. Skip this directory
        # and log this event.
        try:
            model_root = Path(directory)
            with (model_root / server / 'learnedModel.dot').open() as f:
                # This is where the deduplication happens
                models[f.read()].append(server)

                logger.info(f'Found model for: {server}')
        except OSError:
            logger.warning(f'Could not find model for: {server}')

    # Initialize an empty tree with a single node, all models will be merged
    # into this tree.
    tree = networkx.DiGraph()
    root = tuple()
    tree.add_node(root)

    # For each model, convert to a networkx graph and merge into the tree
    for model, servers in models.items():
        # 'graph_from_dot_data()' returns a list, but StateLearner only puts a
        # single graph in a file, we don't have to check the length.
        pydot_graph = pydot.graph_from_dot_data(model)[0]

        # Convert to networkx graph
        graph = networkx.drawing.nx_pydot.from_pydot(pydot_graph)

        # The start node is always 's0'
        tree = _merge_graph(tree, root, graph, 's0', servers)

    return tree
