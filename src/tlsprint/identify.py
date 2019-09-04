"""Identification components, to be used after learning the model tree."""

import os
import pathlib
import random
import socket
import subprocess

import pkg_resources


def probe(connector, message):
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
    connector.sendall((message + "\n").encode())
    return connector.recv(bufsize).decode().strip()


def identify(tree, target, target_port=443, graph_dir=None):
    # Create output directory if required
    if graph_dir:
        graph_dir = pathlib.Path(graph_dir)
        graph_dir.mkdir(exist_ok=True)

    # Start TLSAttackerConnector
    connector_path = pkg_resources.resource_filename(
        __name__, os.path.join("connector", "TLSAttackerConnector2.0.jar")
    )
    connector_process = subprocess.Popen(
        [
            "java",
            "-jar",
            connector_path,
            "--targetHost",
            target,
            "--targetPort",
            str(target_port),
        ],
        stdout=subprocess.PIPE,
    )

    # Wait until the first line to stdout is written, this means the connector
    # is initialized.
    connector_process.stdout.readline()

    # Connect to the connector socket
    connector = socket.create_connection(("localhost", 6666))

    identifing = True
    iteration = 1
    while identifing:

        # Reset TLSAttackerConnector
        probe(connector, "RESET")

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
            response = probe(connector, send_node[-1])

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
    connector_process.terminate()
