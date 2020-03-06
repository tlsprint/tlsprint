"""Identification components, to be used after learning the model tree."""

import abc
import math
import operator
import os
import pathlib
import random
import socket
import subprocess

import pkg_resources


def _tree_weight(tree, model_mapping):
    return sum([len(model_mapping[model]) for model in tree.models])
    # return len(tree.models)


def random_selector(tree, current_node):
    return random.choice(list(tree[current_node]))


def always_first_selector(tree, current_node):
    return list(tree[current_node])[0]


def gini_selector(tree, current_node):
    """Use the Gini Impurity to compute with inputs leads to the most
    distinguishing outputs.
    More information here: https://en.wikipedia.org/wiki/Decision_tree_learning#Metrics
    """
    total_weight = _tree_weight(tree.subtree(current_node), tree.model_mapping)
    input_info = [{"node": node} for node in tree[current_node]]
    for info in input_info:
        output_nodes = list(tree[info["node"]])
        weights = [
            _tree_weight(tree.subtree(output_node), tree.model_mapping)
            for output_node in output_nodes
        ]
        info["metric"] = 1 - sum([(x / total_weight) ** 2 for x in weights])

    return max(input_info, key=operator.itemgetter("metric"))["node"]


def entropy_selector(tree, current_node):
    """Use the Information Gain, based on entropy, to compute with inputs leads
    to the most distinguishing outputs.
    More information here: https://en.wikipedia.org/wiki/Decision_tree_learning#Metrics
    """
    total_weight = _tree_weight(tree.subtree(current_node), tree.model_mapping)
    input_info = [{"node": node} for node in tree[current_node]]
    for info in input_info:
        output_nodes = list(tree[info["node"]])
        weights = [
            _tree_weight(tree.subtree(output_node), tree.model_mapping)
            for output_node in output_nodes
        ]
        info["metric"] = -1 * sum(
            [(x / total_weight) * math.log(x / total_weight) for x in weights]
        )

    return max(input_info, key=operator.itemgetter("metric"))["node"]


INPUT_SELECTORS = {
    "random": random_selector,
    "first": always_first_selector,
    "gini": gini_selector,
    "entropy": entropy_selector,
}


class AbastractConnector(abc.ABC):
    def close(self):
        pass

    @abc.abstractmethod
    def send(self, message):
        pass

    def descent(self, tree, selector, graph_dir=None):
        """Descent the tree until a leaf node is reached."""
        # Start at the root of the tree
        current_node = tuple()

        leaves = tree.leaves
        descending = True
        while descending:
            # Pick a random node (message to send)
            send_node = selector(tree, current_node)

            # Send this message and read the response
            response = self.send(send_node[-1])

            # Check if this leads to an existing node, and if this node is a
            # leaf node.
            response_node = send_node + (response,)
            try:
                tree[response_node]
            except KeyError:
                print("No model with this path:")
                print(response_node)
                return

            if response_node in leaves:
                descending = False
            else:
                current_node = response_node

        return response_node


class TLSAttackerConnector(AbastractConnector):
    def __init__(self, target, target_port=443):
        """Start TLSAttackerConnector. Returns a handler to both the process and
        the socket"""
        connector_path = pkg_resources.resource_filename(
            __name__, os.path.join("connector", "TLSAttackerConnector2.0.jar")
        )
        messages_path = pkg_resources.resource_filename(
            __name__, os.path.join("connector", "messages")
        )

        self.process = subprocess.Popen(
            [
                "java",
                "-jar",
                connector_path,
                "--targetHost",
                target,
                "--targetPort",
                str(target_port),
                "--messageDir",
                messages_path,
                "--merge-application",
            ],
            stdout=subprocess.PIPE,
        )

        # Wait until the first line to stdout is written, this means the connector
        # is initialized.
        self.process.stdout.readline()

        # Connect to the connector socket
        self.socket = socket.create_connection(("localhost", 6666))

    def close(self):
        self.socket.close()
        self.process.terminate()

    def send(self, message):
        """Send the message to TLSAttackerConnector and return the result.

        This function does a few things:
            - Append a newline to the message
            - Encode the message
            - Decodes the resulting response
            - Strips the response of the trailing newline
        """
        # TLSAttackerConnector will never be larger than this, but something more
        # robust is desirable.
        bufsize = 1024
        self.socket.sendall((message + "\n").encode())
        return self.socket.recv(bufsize).decode().strip()

    def reset(self):
        self.send("RESET")


class BenchmarkConnector(AbastractConnector):
    def __init__(self, target, tree):
        self.target = target
        self.tree = tree

        # Initialize a list to keep track of the messages send and received
        self.messages = []
        self.current_node = ()

    def send(self, message):
        self.messages.append(message)
        self.current_node += (message,)

        neighbors = self.tree[self.current_node]
        for neighbor in neighbors:
            if self.target in self.tree.subtree(neighbor).models:
                output = neighbor[-1]
                self.messages.append(output)
                self.current_node += (output,)
                return output

    def reset(self):
        self.messages += ["RESET", ""]
        self.current_node = ()


def identify(
    tree,
    target,
    target_port=443,
    graph_dir=None,
    selector=always_first_selector,
    benchmark=False,
):
    # Create output directory if required
    if graph_dir:
        graph_dir = pathlib.Path(graph_dir)
        graph_dir.mkdir(exist_ok=True)

    if benchmark:
        connector = BenchmarkConnector(target, tree)
    else:
        connector = TLSAttackerConnector(target, target_port)

    identifing = True
    iteration = 1
    while identifing:

        # Descent to a leaf node
        leaf_node = connector.descent(tree, selector, graph_dir=graph_dir)

        # If the descent does not return a leaf node, there is no model
        # matched.
        if not leaf_node:
            connector.close()
            return

        # Log the node, color it and draw the graph
        if graph_dir:
            tree.nodes[leaf_node]["color"] = "red"
            tree.draw(
                path=graph_dir / "iteration-{}-pre-prune.svg".format(iteration),
                fmt="svg",
            )

        # Prune the tree
        leaf_models = tree.nodes[leaf_node]["models"]
        tree.prune_models(tree.models - leaf_models)

        if graph_dir:
            tree.draw(
                path=graph_dir / "iteration-{}-post-prune.svg".format(iteration),
                fmt="svg",
            )

        # Condense the tree
        tree.condense()

        # If the tree is empty after condensing, the result was one of the
        # models in the last leaf node. This can be more then one model, as
        # some might not be distinguishable.
        if len(tree) == 0:
            connector.close()
            if benchmark:
                return connector.messages
            else:
                return leaf_models

        if graph_dir:
            tree.draw(
                path=graph_dir / "iteration-{}-condensed.svg".format(iteration),
                fmt="svg",
            )

        iteration += 1

        # Reset TLSAttackerConnector
        connector.reset()
