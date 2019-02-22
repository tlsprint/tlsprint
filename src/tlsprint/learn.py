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
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import pydot


def _append_servers(tree, node, servers):
    try:
        tree.nodes[node]["servers"]
    except KeyError:
        tree.nodes[node]["servers"] = set()
    tree.nodes[node]["servers"].update(servers)


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

            received_node = _merge_path_from_label(tree, root, edge["label"])

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
                sent, received = (
                    x.replace('"', "").strip() for x in edge["label"].split("/")
                )

                # Append the sent and received messages to the tree
                sent_node = root + (sent,)
                received_node = root + (sent, received)
                tree.add_edge(root, sent_node, label=sent)
                tree.add_edge(sent_node, received_node, label=received)

                # If the received message contains 'ConnectionClosed', this
                # path can be stopped here. This greatly reduces the number
                # of redundant nodes, because of 'ConnectionClosed' edges
                # go the the final node, which always contains many self loops.
                if "ConnectionClosed" in received:
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
                found_loop = False
                cycles = list(simple_cycles(graph))
                for cycle in cycles:
                    if current_node == cycle[-1]:
                        loop_edge = graph[current_node][cycle[0]][0]
                        sent, received = [
                            x.replace('"', "").strip()
                            for x in loop_edge["label"].split("/")
                        ]

                        loop_cause_node = received_node + (sent,)
                        loop_node = loop_cause_node + ("CYCLE",)
                        tree.add_edge(received_node, loop_cause_node, label=sent)
                        tree.add_edge(loop_cause_node, loop_node, label="CYCLE")

                        # Append the servers
                        _append_servers(tree, loop_node, servers)

                        # Do not recurse
                        found_loop = True

                if not found_loop:
                    # Recurse with new root and current node
                    tree = _merge_graph(tree, received_node, graph, neighbor, servers)

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

    # Merge duplicates together, by using the DOT model as dictionary key,
    # and the server list as the value.
    models = defaultdict(list)

    for server in server_dirs:
        # Not all directories actually contain a model. Skip this directory
        # and log this event.
        try:
            model_root = Path(directory)
            with (model_root / server / "learnedModel.dot").open() as f:
                # This is where the deduplication happens
                models[f.read()].append(server)

                logger.info("Found model for: {}".format(server))
        except OSError:
            logger.warning("Could not find model for: {}".format(server))

    # Initialize an empty tree with a single node, all models will be merged
    # into this tree.
    tree = ModelTree()
    root = tuple()
    tree.add_node(root)

    # For identical models, store a group name instead of the list and create a
    # mapping between the name and the implementations. This will reduce the
    # size of the tree when drawn, and will no longer cause confusing as to
    # which implementations have the same model. This mapping will be stored in
    # the ModelTree
    tree.model_mapping = {}
    for i, (model, servers) in enumerate(models.items()):
        name = "model-{}".format(i)
        tree.model_mapping[name] = servers
        models[model] = [name]

    # For each model, convert to a networkx graph and merge into the tree
    for model, servers in models.items():
        # 'graph_from_dot_data()' returns a list, but StateLearner only puts a
        # single graph in a file, we don't have to check the length.
        pydot_graph = pydot.graph_from_dot_data(model)[0]

        # Convert to networkx graph
        graph = networkx.drawing.nx_pydot.from_pydot(pydot_graph)

        # The start node is always 's0'
        tree = _merge_graph(tree, root, graph, "s0", servers)

    return tree


class ModelTree(networkx.DiGraph):
    @property
    def leaves(self):
        return [node for node in self.nodes if self.out_degree(node) == 0]

    @property
    def groups(self):
        return {group for leaf in self.leaves for group in self.nodes[leaf]["servers"]}

    def subtree(self, node):
        """Return the subtree where `node` is the root, as a ModelTree."""
        subtree_nodes = dfs_tree(self, node).nodes
        return self.subgraph(subtree_nodes)

    def prune_node(self, node):
        """Cut a node from the tree, pruning the predecessors away as far as
        possible.
        """
        redundant_nodes = [node]

        pruning = True
        while pruning:
            node = redundant_nodes[-1]
            # It is a tree, to each node only has one predecessor at
            # most.
            try:
                predecessor = list(self.predecessors(node))[0]
            except IndexError:
                break  # At the top of the tree

            if self.out_degree(predecessor) == 1:
                # Only connected to the lower redundant node
                redundant_nodes.append(predecessor)
            else:
                pruning = False

        # Remove the redundant nodes
        for node in redundant_nodes:
            self.remove_node(node)

    def prune_groups(self, groups):
        """Prune the specified groups from the tree, removing redundant nodes
        from the tree."""
        groups = set(groups)

        for leaf in self.leaves:
            # Start by removing the groups from every leaf node
            self.nodes[leaf]["servers"] -= groups

            # If the set is non empty, we can remove it and also their
            # predecessors if they are only connected to this leaf.
            if not self.nodes[leaf]["servers"]:
                self.prune_node(leaf)

    def condense(self):
        """Make the tree more compact by merging together nodes whose leaves
        are all the same, and the leaves that contain 100% of the groups in the
        model."""
        # Remove leafs that contain 100% of the groups
        groups = self.groups
        for leaf in self.leaves:
            if self.nodes[leaf]["servers"] == groups:
                self.prune_node(leaf)

        # For each leaf, check if its parent contains other groups. If not,
        # merge them together in one node.
        changed = True
        while changed:
            changed = False

            for leaf in self.leaves:
                # Take the subtree of two predecessors up (because of the
                # structure of the graph)
                one_up = list(self.predecessors(leaf))[0]
                two_up = list(self.predecessors(one_up))[0]

                subtree = self.subtree(two_up)
                groups = self.nodes[leaf]["servers"]

                if subtree.groups == groups and subtree.out_degree(two_up) == len(
                    subtree.leaves
                ):
                    # List the servers at this root node in the original tree,
                    # and remove the other nodes
                    self.nodes[two_up]["servers"] = groups
                    redundant_nodes = set(subtree.nodes) - {two_up}
                    self.remove_nodes_from(redundant_nodes)
                    changed = True
                    break

    def draw(self, path):
        """Draw this tree by outputting a Graphviz file in DOT format. This
        slightly modifies the tree in order to improve the output:
        -   Set the label of all non leafs nodes to blank, as the information
            is already captured by the edges.
        -   Set the label of all leaf nodes to the list of servers, including
            the percentage of how much servers are contained in this leaf,
            compared to all present in the tree.

        Args:
            tree: The tree to modify and draw.
            path: The path where to store the DOT file.
        """
        group_count = len(self.groups)

        # Relabel all the nodes
        for node in self.nodes:
            if self.out_degree(node) == 0:
                # Leaf node
                groups = sorted(self.nodes[node]["servers"])
                group_share = "{:.2f}%".format(100 * len(groups) / group_count)
                self.nodes[node]["label"] = "\n".join([group_share] + groups)
                self.nodes[node]["shape"] = "rectangle"
            else:
                # Not a leaf node
                self.nodes[node]["label"] = ""
        networkx.drawing.nx_pydot.write_dot(self, path)


def normalize_graph(dot_graph: str) -> ModelTree:
    """Normalizes an input graph into a ModelTree. It is possible that an input
    graph has multiple DOT representations (think of whitespace differences,
    but also graphs that have the same structure but different node names. In
    order to see if these are identical, we normalize them.

    Args:
        dot_graph: The DOT representation of the input graph

    Returns:
        A normalized ModelTree which represents the input graph.
    """
    graph = _dot_to_networkx(dot_graph)

    # Assumes there is a node called '__start0', which is connected a single
    # node in the graph (the entry point)
    graph_root = list(graph["__start0"])[0]

    # Create the ModelTree that will contain the normalized graph
    tree = ModelTree()
    tree_root = ()
    tree.add_node(tree_root)

    # Normalize the graph by recursively merging in into the tree
    return _merge_subgraph(tree, tree_root, graph, graph_root)


def _merge_subgraph(
    tree: ModelTree, root: tuple, graph: networkx.DiGraph, current_node: str
) -> ModelTree:
    """Recursively merge a directed graph into the passed ModelTree. This is an
    internal function that is called from `normalize_graph`.

    Args:
        tree: The tree that is being constructed, and into which the graph will
            be merged.
        root: The current root node of the tree, the graph will be merged
            beginning at this node.
        graph: The graph to merge into the tree
        current_node: The current node of the graph to merge
    """
    neighbors = list(graph[current_node])

    # If a node has no neighbors, there is nothing to do and this function
    # returns immediately.
    if not neighbors:
        return tree

    # If a node has itself as its only neighbor, then this is a final node, and
    # we should not recurse (or this will lead to infinite loops. Simple add
    # its neighbors and return.
    if neighbors == [current_node]:
        for _, edge in graph[current_node][current_node].items():
            received_node = _merge_path_from_label(tree, root, edge["label"])
        return tree

    # In any other case, follow every edge and recursively merge the rest of
    # the graph into the tree.

    # Find cycles in this graph, this is used later in the loop, but we only
    # have to do it once
    cycles = list(simple_cycles(graph))

    # A node can have multiple neighbors
    for neighbor in neighbors:
        # There can be multiple edges between two nodes, each with
        # a different label. Each edge is numbered, but we ignore this
        # number.
        for _, edge in graph[current_node][neighbor].items():
            # We start by checking if the current node and its neighbor is part
            # of a cycle. This is the case if the current node if the final
            # node, and its neighbors is the first node of any cycle (if we do
            # not check this, we can end up in an infinite loop). To indicate
            # cycles in the normalized graph, we set a `postfix` value, which
            # is passed to the `_merge_path_from_label` function.
            cycle_detected = False
            postfix = None
            for cycle in cycles:
                if current_node == cycle[-1] and neighbor == cycle[0]:
                    cycle_detected = True
                    cycle_path = _extract_cycle_path(graph, cycle)
                    postfix = "|" + cycle_path

            received_node = _merge_path_from_label(
                tree, root, edge["label"], postfix=postfix
            )

            # If the received message contains 'ConnectionClosed', this
            # path can be stopped here. This greatly reduces the number
            # of redundant nodes, because of 'ConnectionClosed' edges
            # go the the final node, which always contains many self loops.
            if "ConnectionClosed" in received_node[-1]:
                # Do not recurse
                continue

            # Recurse with new root and current node
            if not cycle_detected:
                tree = _merge_subgraph(tree, received_node, graph, neighbor)

    return tree


def _merge_path_from_label(
    tree: ModelTree, root: tuple, label: str, postfix: str = None
) -> tuple:
    """Merge a path into the passed tree from a label. The label is assumed to
    have the format "{{ sent }} / {{ received }}", since this is the format
    that StateLearner outputs. The nodes will be added as

        root -> sent -> received

    with the appropriate edge labels.

    Args:
        tree: The path will be added to this graph.
        root: Point in the tree where the nodes will be added.
        label: String of the format "{{ sent }} / {{ received }}"
        postfix: Optional value that will be added to the received node name.

    Returns:
        The name of the "received" node (which is a tuple), so the caller knows
        the endpoint of the added path.
    """
    # We start by extracting the sent and received messages. Split the label
    # in the sent and received message. Remove the double quotes and the excess
    # whitespace.
    sent, received = [
        message.replace('"', "").strip() for message in label.split("/", maxsplit=1)
    ]

    if postfix:
        received += postfix

    # Append the sent and received messages to the tree
    sent_node = root + (sent,)
    received_node = root + (sent, received)
    tree.add_edge(root, sent_node, label=sent)
    tree.add_edge(sent_node, received_node, label=received)

    return received_node


def _dot_to_networkx(dot_graph):
    """Convert a DOT string to a networkx graph."""
    # Read the input graph using `graph_from_dot_data()`. This function returns
    # a list but StateLearner only puts a single graph in a file. We assume
    # this this graph is present and do not check the length. A KeyError will
    # notify us in case of an error.
    pydot_graph = pydot.graph_from_dot_data(dot_graph)[0]

    # Convert to networkx graph
    return networkx.drawing.nx_pydot.from_pydot(pydot_graph)


def _extract_cycle_path(graph, cycle):
    cycle = cycle + [cycle[0]]
    edges = list(zip(cycle, cycle[1:]))

    # We assume there is only one edge between all the nodes in the cycle
    labels = [graph[edge[0]][edge[1]][0]["label"] for edge in edges]
    return "CYCLE({})".format(" -> ".join(labels))
