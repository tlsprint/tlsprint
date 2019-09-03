from tlsprint.learn import normalize_graph


def test_one_edge():
    dot_graph = """digraph {
        __start0 -> a
        a -> b [label="A / B"]
    }"""
    tree = normalize_graph(dot_graph)
    expected = {(), ("A",), ("A", "B")}
    assert expected == set(tree.nodes)


def test_multiple_edges():
    dot_graph = """digraph {
        __start0 -> a
        a -> b [label="A / B"]
        a -> c [label="A / C"]
        b -> c [label="B / C"]
    }"""
    tree = normalize_graph(dot_graph)
    expected = {
        (),
        ("A",),
        ("A", "C"),
        ("A", "B"),
        ("A", "B", "B"),
        ("A", "B", "B", "C"),
    }
    assert expected == set(tree.nodes)


def test_multiple_edges_different_structure():
    dot_graph = """digraph graph_name {
        __start0 -> name1
        name1 -> name2 [label="A / B"]
        name2 -> name3 [label="B / C"]
        name1 -> name3 [label="A / C"]
    }"""
    tree = normalize_graph(dot_graph)
    expected = {
        (),
        ("A",),
        ("A", "C"),
        ("A", "B"),
        ("A", "B", "B"),
        ("A", "B", "B", "C"),
    }
    assert expected == set(tree.nodes)


def test_final_node_loop():
    dot_graph = """digraph {
        __start0 -> a
        a -> b [label="A / B"]
        b -> b [label="A / C"]
        b -> b [label="D / B"]
    }"""
    tree = normalize_graph(dot_graph)
    expected = {
        (),
        ("A",),
        ("A", "B"),
        ("A", "B", "A"),
        ("A", "B", "A", "C"),
        ("A", "B", "D"),
        ("A", "B", "D", "B"),
    }
    assert expected == set(tree.nodes)


def test_connection_closed():
    dot_graph = """digraph {
        __start0 -> a
        a -> b [label="A / B"]
        b -> b [label="A / C"]
        a -> b [label="D / B|ConnectionClosed"]
    }"""
    tree = normalize_graph(dot_graph)
    expected = {
        (),
        ("A",),
        ("A", "B"),
        ("A", "B", "A"),
        ("A", "B", "A", "C"),
        ("D",),
        ("D", "B|ConnectionClosed"),
    }
    assert expected == set(tree.nodes)


def test_single_node_cycle():
    dot_graph = """digraph {
        __start0 -> a
        a -> b [label="A / B"]
        b -> c [label="C / D"]
        b -> b [label="A / E"]
    }"""
    tree = normalize_graph(dot_graph, max_depth=3)
    expected = {
        (),
        ("A",),
        ("A", "B"),
        ("A", "B", "C"),
        ("A", "B", "C", "D"),
        ("A", "B", "A"),
        ("A", "B", "A", "E"),
        ("A", "B", "A", "E", "C"),
        ("A", "B", "A", "E", "C", "D"),
        ("A", "B", "A", "E", "A"),
        ("A", "B", "A", "E", "A", "E"),
        ("A", "B", "A", "E", "A", "E", "A"),
        ("A", "B", "A", "E", "A", "E", "A", "E"),
        ("A", "B", "A", "E", "A", "E", "C"),
        ("A", "B", "A", "E", "A", "E", "C", "D"),
    }
    assert expected == set(tree.nodes)


def test_multi_node_cycle():
    dot_graph = """digraph {
        __start0 -> a
        a -> b [label="A / B"]
        b -> c [label="C / D"]
        c -> d [label="F / G"]
        c -> b [label="A / E"]
    }"""
    tree = normalize_graph(dot_graph, max_depth=4)
    expected = {
        (),
        ("A",),
        ("A", "B"),
        ("A", "B", "C"),
        ("A", "B", "C", "D"),
        ("A", "B", "C", "D", "F"),
        ("A", "B", "C", "D", "F", "G"),
        ("A", "B", "C", "D", "A"),
        ("A", "B", "C", "D", "A", "E"),
        ("A", "B", "C", "D", "A", "E", "C"),
        ("A", "B", "C", "D", "A", "E", "C", "D"),
        ("A", "B", "C", "D", "A", "E", "C", "D", "A"),
        ("A", "B", "C", "D", "A", "E", "C", "D", "A", "E"),
        ("A", "B", "C", "D", "A", "E", "C", "D", "F"),
        ("A", "B", "C", "D", "A", "E", "C", "D", "F", "G"),
    }
    assert expected == set(tree.nodes)
