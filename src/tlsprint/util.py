import networkx


def prefix_nodes(graph, prefix):
    """Prefix every node of the graph with the specified prefix."""
    mapping = {node: f"{prefix}{node}" for node in graph.nodes}
    return networkx.relabel_nodes(graph, mapping)


def add_resets(graph, start):
    """Add a "RESET / -" edge from every sinkhole to the start state."""
    for node in graph.nodes:
        neighbors = list(graph[node])
        if neighbors == [node]:
            graph.add_edge(node, start, label="RESET / -")


def convert_graph(graph, *, add_resets=False):
    """Convert a graph from LearnLib DOT output to dict, with the structure
    required by adg-finder.
    """
    converted = {}

    # The first (and only) state connected to the dummy_start, is the actual
    # start state.
    dummy_start = [node for node in graph.nodes if graph.in_degree(node) == 0][0]
    converted["initial_state"] = [node for node in graph[dummy_start]][0]

    # Remove the dummy start node
    graph.remove_node(dummy_start)

    # Include a sorted list of states
    converted["states"] = sorted(graph.nodes)

    # If requested, add reset edges
    if add_resets:
        add_resets(graph, converted["initial_state"])

    # Convert all transitions to tuple format
    converted["transitions"] = []
    inputs = set()
    outputs = set()
    for edge in graph.edges:
        label = graph.edges[edge]["label"]
        sent, received = (x.replace('"', "").strip() for x in label.split("/"))

        source, destination, _ = edge
        transition = [(source, sent), (received, destination)]
        converted["transitions"].append(transition)

        inputs.add(sent)
        outputs.add(received)

    converted["inputs"] = sorted(inputs)
    converted["outputs"] = sorted(outputs)

    return converted
