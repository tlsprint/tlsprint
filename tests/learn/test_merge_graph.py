import networkx
from tlsprint.learn import _merge_graph


def test_end_condition_simple():
    # Create tree with empty start node
    tree = networkx.DiGraph()
    root = tuple()
    tree.add_node(root)

    # Create graph with single node with self loop, with the same
    # structure as graphs produced by pydot
    graph = networkx.DiGraph([("s2", "s2", {0: {"label": "sent / received"}})])

    # Merge the graph into the tree
    servers = ["serverA", "serverB"]
    tree = _merge_graph(tree, root, graph, "s2", servers)

    # Assert that the tree is correct
    sent = ("sent",)
    received = ("sent", "received")
    assert set(tree.nodes) == {root, sent, received}
    assert set(tree.edges) == {(root, sent), (sent, received)}
    assert dict(tree[root][sent]) == {"label": "sent"}
    assert dict(tree[sent][received]) == {"label": "received"}
    assert tree.nodes[received]["servers"] == set(servers)


def test_recursion():
    # Create tree with empty start node
    tree = networkx.DiGraph()
    root = tuple()
    tree.add_node(root)

    # Create graph with multiple nodes, where the end node has a self loop,
    # with the same structure as graphs produced by pydot
    graph = networkx.DiGraph(
        [
            ("s1", "s2", {0: {"label": "sentA / receivedA"}}),
            ("s2", "s2", {0: {"label": "sentB / receivedB"}}),
        ]
    )

    # Merge the graph into the tree
    servers = ["serverA", "serverB"]
    tree = _merge_graph(tree, root, graph, "s1", servers)

    # Assert that the tree is correct
    sentA = ("sentA",)
    receivedA = ("sentA", "receivedA")
    sentB = ("sentA", "receivedA", "sentB")
    receivedB = ("sentA", "receivedA", "sentB", "receivedB")
    assert set(tree.nodes) == {root, sentA, receivedA, sentB, receivedB}
    assert set(tree.edges) == {
        (root, sentA),
        (sentA, receivedA),
        (receivedA, sentB),
        (sentB, receivedB),
    }
    assert dict(tree[root][sentA]) == {"label": "sentA"}
    assert dict(tree[sentA][receivedA]) == {"label": "receivedA"}
    assert dict(tree[receivedA][sentB]) == {"label": "sentB"}
    assert dict(tree[sentB][receivedB]) == {"label": "receivedB"}
    assert tree.nodes[receivedB]["servers"] == set(servers)
