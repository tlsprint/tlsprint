"""The learning component of tlsprint. The functions in this module can learn
from the output of StateLearning and create a model of all TLS implementations
in order to perform fingerprinting.
"""

import ast
import json
from pathlib import Path

import networkx
import pydot
from networkx.algorithms.traversal.depth_first_search import dfs_tree


class ModelTree(networkx.DiGraph):
    """Data structure to store an ADG or HDT created from LearnLib models."""

    def parent(self, node):
        """Return the parent of the specified node."""
        # This uses the `predecessors` function, with the assumption that
        # each node has at most one predecessors, it's parent.
        return list(self.predecessors(node))[0]

    @property
    def leaves(self):
        return [node for node in self.nodes if self.out_degree(node) == 0]

    @property
    def models(self):
        return {
            _models for leaf in self.leaves for _models in self.nodes[leaf]["models"]
        }

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
            try:
                parent = self.parent(node)
            except IndexError:
                break  # At the top of the tree

            if self.out_degree(parent) == 1:
                # Only connected to the lower redundant node
                redundant_nodes.append(parent)
            else:
                pruning = False

        # Remove the redundant nodes
        for node in redundant_nodes:
            self.remove_node(node)

    def prune_models(self, models):
        """Prune the specified models from the tree, removing redundant nodes
        from the tree."""
        models = set(models)

        for leaf in self.leaves:
            # Start by removing the models from every leaf node
            self.nodes[leaf]["models"] -= models

            # If the set is non empty, we can remove it and also their
            # predecessors if they are only connected to this leaf.
            if not self.nodes[leaf]["models"]:
                self.prune_node(leaf)

    def condense(self):
        """Make the tree more compact by merging together nodes whose leaves
        are all the same, and the leaves that contain 100% of the models in the
        tree."""
        # Remove leafs that contain 100% of the models
        models = self.models
        for leaf in self.leaves:
            if self.nodes[leaf]["models"] == models:
                self.prune_node(leaf)

        # For each leaf, check if its parent contains other models. If not,
        # merge them together in one node.
        changed = True
        while changed:
            changed = False

            for leaf in self.leaves:
                # Take the subtree of two predecessors up (because of the
                # structure of the graph)
                one_up = self.parent(leaf)
                two_up = self.parent(one_up)

                subtree = self.subtree(two_up)
                models = self.nodes[leaf]["models"]

                if subtree.models == models and subtree.out_degree(two_up) == len(
                    subtree.leaves
                ):
                    # List the models at this root node in the original tree,
                    # and remove the other nodes
                    self.nodes[two_up]["models"] = models
                    redundant_nodes = set(subtree.nodes) - {two_up}
                    self.remove_nodes_from(redundant_nodes)
                    changed = True
                    break

    def draw(self, fmt="dot", path=None):
        """Draw this tree using Graphviz in a desired output format. This
        slightly modifies the tree in order to improve the output:
        -   Set the label of all non leafs nodes to blank, as the information
            is already captured by the edges.
        -   Set the label of all leaf nodes to the list of servers, including
            the percentage of how much servers are contained in this leaf,
            compared to all present in the tree.

        Args:
            tree: The tree to modify and draw.
            fmt: Any format supported by Graphviz in which to draw to graph.
        """
        try:
            model_count = len(self.models)
        except KeyError:
            pass

        # Relabel all the nodes
        for node in self.nodes:
            if self.out_degree(node) == 0:
                # Leaf node
                try:
                    models = sorted(self.nodes[node]["models"])
                    model_share = "{:.2f}%".format(100 * len(models) / model_count)
                    self.nodes[node]["label"] = "\n".join([model_share] + models)
                except KeyError:
                    self.nodes[node]["label"] = ""
                self.nodes[node]["shape"] = "rectangle"
            else:
                # Not a leaf node
                self.nodes[node]["label"] = ""
        dot = networkx.drawing.nx_pydot.to_pydot(self)
        result = dot.create(format=fmt)

        if path:
            with open(path, "wb") as file:
                file.write(result)

        return dot.create(format=fmt)


def normalize_graph(dot_graph: str, *, max_depth=10) -> ModelTree:
    """Normalizes an input graph into a ModelTree. It is possible that an input
    graph has multiple DOT representations (think of whitespace differences,
    but also graphs that have the same structure but different node names. In
    order to see if these are identical, we normalize them by unwrapping them
    as a tree.

    Args:
        dot_graph: The DOT representation of the input graph
        max_depth: The maximum depth of the tree, especially relevant when the
            graph contains cycles.

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

    # Normalize the graph by recursively merging into the tree
    return _merge_subgraph(tree, tree_root, graph, graph_root, 0, max_depth)


def _merge_subgraph(
    tree: ModelTree,
    root: tuple,
    graph: networkx.DiGraph,
    current_node: str,
    current_depth: int,
    max_depth: int,
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
        current_depth: The current recursion depth, this function aborts when
            depth > max_depth.
        max_depth: The maximum recursion depth, primaraly useful to escape
            cycles, so it should be higher than the valid depth of the tree.
    """
    # If we exceeded the max depth, we stop
    if current_depth > max_depth:
        return tree

    neighbors = list(graph[current_node])

    # If a node has no neighbors, there is nothing to do and this function
    # returns immediately.
    if not neighbors:
        return tree

    # A node can have multiple neighbors
    for neighbor in neighbors:

        # There can be multiple edges between two nodes, each with
        # a different label. Each edge is numbered, but we ignore this
        # number.
        for _, edge in graph[current_node][neighbor].items():
            received_node = _merge_path_from_label(tree, root, edge["label"])

            # If the received message contains 'ConnectionClosed', this
            # path can be stopped here. This greatly reduces the number
            # of redundant nodes, because of 'ConnectionClosed' edges
            # go to the final node, which always contains many self loops.
            if "ConnectionClosed" in received_node[-1]:
                # Do not recurse
                continue

            # If a node only has itself as a neighbor, this is a sink state and
            # we do not recurse
            if neighbors == [current_node]:
                continue

            # Recurse with new root and current node
            tree = _merge_subgraph(
                tree, received_node, graph, neighbor, current_depth + 1, max_depth
            )

    return tree


def _merge_path_from_label(tree: ModelTree, root: tuple, label: str) -> tuple:
    """Merge a path into the passed tree from a label. The label is assumed to
    have the format "{{ sent }} / {{ received }}", since this is the format
    that StateLearner outputs. The nodes will be added as

        root -> sent -> received

    with the appropriate edge labels.

    Args:
        tree: The path will be added to this graph.
        root: Point in the tree where the nodes will be added.
        label: String of the format "{{ sent }} / {{ received }}"

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


def construct_tree_from_dedup(directory: str, tree_type: str) -> ModelTree:
    """Given a directory output from the dedup command, construct a ModelTree.

    Args:
        directory: The path to the dedup directory
        tree_type: The desired output tree type, can be any from
            SUPPORTED_TREE_TYPES.
    """
    try:
        handler = _tree_type_handlers[tree_type]
    except KeyError:
        raise ValueError(f"Not a valid tree type: {tree_type}")

    path = Path(directory)

    # Build the tree using the specified tree type handler
    tree = handler(path)

    # Add the model mapping information to the tree
    tree.model_mapping = {}

    model_directories = sorted([item for item in path.iterdir() if item.is_dir()])
    for model_dir in model_directories:
        with open(model_dir / "versions.json") as f:
            version_info = json.load(f)

            # Convert to set with tuples and add to model_mapping
            version_info = {tuple(x) for x in version_info}
            tree.model_mapping[model_dir.name] = version_info

    return tree


def _construct_adg(path: Path) -> ModelTree:
    """Construct the ADG (output from adg-finder) and add metadata from the
    dedup directory.
    """
    adg_path = path / "adg.gv"
    with open(adg_path) as f:
        adg = _dot_to_networkx(f.read())
    adg_root = [node for node in adg.nodes if adg.in_degree(node) == 0][0]

    tree = ModelTree()
    tree_root = ()
    tree.add_node(tree_root)

    return _merge_subadg(tree, tree_root, adg, adg_root)


def _merge_subadg(tree, root, adg, current_node):
    neighbors = list(adg[current_node])

    if not neighbors:
        node = adg.nodes[current_node]

        # Fix adg-finder output (add brackets, remove quotes, remove "_s0"
        # suffixes).
        models = f"[{node['models']}]"
        models = models.replace('"', "")
        models = models.replace("_s0", "")

        # Parse as Python list
        models = ast.literal_eval(models)

        # Add as attribute to node
        tree.nodes[root]["models"] = set(models)

        return tree

    for neighbor in neighbors:
        for _, edge in adg[current_node][neighbor].items():
            label = edge["label"].replace('"', "")
            new_node = root + (label,)

            tree.add_edge(root, new_node, label=label)

            # Recurse
            tree = _merge_subadg(tree, new_node, adg, neighbor)

    return tree


def _construct_hdt(path: Path) -> ModelTree:
    """Construct the HDT (heuristic decision tree) from the dedup
    directory.
    """
    tree = ModelTree()
    tree_root = ()
    tree.add_node(tree_root)

    model_directories = sorted([item for item in path.iterdir() if item.is_dir()])
    for model_dir in model_directories:
        with open(model_dir / "model.gv") as f:
            graph = normalize_graph(f.read())
            tree.add_edges_from(graph.edges(data=True))
            for leaf in graph.leaves:
                try:
                    tree.nodes[leaf]["models"].add(model_dir.name)
                except KeyError:
                    tree.nodes[leaf]["models"] = {model_dir.name}

    tree.condense()
    return tree


_tree_type_handlers = {"adg": _construct_adg, "hdt": _construct_hdt}
SUPPORTED_TREE_TYPES = list(_tree_type_handlers.keys())
