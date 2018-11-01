import networkx

from tlsprint.learn import _merge_graph


def test_end_condition_simple():
    # Create tree with empty start node
    tree = networkx.DiGraph()
    tree.add_node(tuple())

    # Create graph with single node with self loop, with the same
    # structure as graphs parsed by pydot
    graph = networkx.DiGraph([('s2', 's2', {
        0: {'label': 'sent / received'}
    })])

    # Merge the graph into the tree
    servers = ['serverA', 'serverB']
    tree = _merge_graph(tree, tuple(), graph, 's2', servers)

    # Assert that the tree is correct
    assert set(tree.nodes) == {
        tuple(),
        ('sent', ),
        ('sent', 'received'),
    }
    assert set(tree.edges) == {
        (tuple(), ('sent', )),
        (('sent', ), ('sent', 'received')),
    }
    assert dict(tree[tuple()][('sent', )]) == {
        'label': 'sent',
    }
    assert dict(tree[('sent', )][('sent', 'received')]) == {
        'label': 'received',
    }
    assert tree.nodes[('sent', 'received')]['servers'] == set(servers)
