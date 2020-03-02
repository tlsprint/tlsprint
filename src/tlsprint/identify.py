"""Identification components, to be used after learning the model tree."""

import abc
import os
import pathlib
import random
import socket
import subprocess

import pkg_resources


class AbastractConnector(abc.ABC):
    def close(self):
        pass

    @abc.abstractmethod
    def send(self, message):
        pass

    def descent(self, tree, graph_dir=None):
        """Descent the tree until a leaf node is reached."""
        # Start at the root of the tree
        current_node = tuple()

        leaves = tree.leaves
        descending = True
        while descending:
            # Pick a random node (message to send)
            send_node = random.choice(list(tree[current_node]))

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


def identify(tree, target, target_port=443, graph_dir=None, benchmark=False):
    # Create output directory if required
    if graph_dir:
        graph_dir = pathlib.Path(graph_dir)
        graph_dir.mkdir(exist_ok=True)

    connector = TLSAttackerConnector(target, target_port)

    identifing = True
    iteration = 1
    while identifing:

        # Reset TLSAttackerConnector
        connector.send("RESET")

        # Descent to a leaf node
        leaf_node = connector.descent(tree, graph_dir)

        # If the descent does not return a leaf node, there is no model
        # matched.
        if not leaf_node:
            connector.close()
            return

        # Log the node, color it and draw the graph
        if graph_dir:
            tree.nodes[leaf_node]["color"] = "red"
            tree.draw(graph_dir / "iteration-{}-pre-prune.gv".format(iteration))

        # If there is only one model left in the leaf node, we have a result.
        leaf_models = tree.nodes[leaf_node]["models"]
        if len(leaf_models) == 1:
            connector.close()
            return leaf_models

        # Prune the tree
        tree.prune_models(tree.models - leaf_models)

        if graph_dir:
            tree.draw(graph_dir / "iteration-{}-post-prune.gv".format(iteration))

        # Condense the tree
        tree.condense()

        if graph_dir:
            tree.draw(graph_dir / "iteration-{}-condensed.gv".format(iteration))
