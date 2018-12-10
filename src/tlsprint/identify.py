"""Identification components, to be used after learning the model tree."""

import random
import socket


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
    connector.sendall((message + '\n').encode())
    return connector.recv(bufsize).decode().strip()


def identify(tree):
    connector = socket.create_connection(('localhost', 6666))

    identifing = True
    iteration = 1
    while identifing:

        # Reset TLSAttackerConnector
        probe(connector, 'RESET')

        # Start at the root of the tree
        current_node = tuple()

        # Pick a random path down the tree
        leaves = tree.leaves
        groups = tree.groups
        descending = True
        while descending:
            # Pick a random node (message to send)
            send_node = random.choice(list(tree[current_node]))

            # Send this message and read the response
            response = probe(connector, send_node[-1])

            # Check if this leads to an existing node, and if this node is a
            # leaf node.
            response_node = send_node + (response, )
            try:
                tree[response_node]
            except KeyError:
                print('No model with this path:')
                print(response_node)
                return

            if response_node in leaves:
                # Log the node, color it and draw the graph
                print(response_node)
                tree.nodes[response_node]['color'] = 'red'
                tree.draw(f'iteration-{iteration}-pre-prune.gv')

                leaf_groups = tree.nodes[response_node]['servers']
                tree.prune_groups(groups - leaf_groups)
                tree.draw(f'iteration-{iteration}-post-prune.gv')
                descending = False
            else:
                current_node = response_node

        # If 'groups' is empty after condensing, we are done return the groups
        # from right before this step
        groups = tree.groups
        tree.condense()
        tree.draw(f'iteration-{iteration}-condensed.gv')

        if not tree.groups:
            connector.close()
            return groups

        iteration += 1
