"""Identification components, to be used after learning the model tree."""

import os
import pathlib
import random
import socket
import subprocess

import pkg_resources


class TLSAttackerConnector:
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


def identify(tree, target, target_port=443, graph_dir=None):
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

        # Start at the root of the tree
        current_node = tuple()

        # Pick a random path down the tree
        leaves = tree.leaves
        models = tree.models
        descending = True
        while descending:
            # Pick a random node (message to send)
            send_node = random.choice(list(tree[current_node]))

            # Send this message and read the response
            response = connector.send(send_node[-1])

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
                # Log the node, color it and draw the graph
                tree.nodes[response_node]["color"] = "red"

                if graph_dir:
                    tree.draw(graph_dir / "iteration-{}-pre-prune.gv".format(iteration))

                leaf_models = tree.nodes[response_node]["models"]
                tree.prune_models(models - leaf_models)

                if graph_dir:
                    tree.draw(
                        graph_dir / "iteration-{}-post-prune.gv".format(iteration)
                    )

                descending = False
            else:
                current_node = response_node

        # If 'models' is empty after condensing, we are done return the models
        # from right before this step
        models = tree.models
        tree.condense()

        if graph_dir:
            tree.draw(graph_dir / "iteration-{}-condensed.gv".format(iteration))

        if not tree.models:
            connector.close()
            return models

        iteration += 1

    # Close the socket and ensure the process is terminated
    connector.close()
