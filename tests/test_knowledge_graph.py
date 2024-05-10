import pytest
from precisely import assert_that, contains_exactly

from knowledge_graph.traverse import Node, Relation

from .conftest import DataFixture


def test_traverse_marie_curie(marie_curie: DataFixture) -> None:
    (result_nodes, result_edges) = marie_curie.graph_store.graph.subgraph(
        start=Node("Marie Curie", "Person"),
        steps=1,
    )
    expected_nodes = [
        Node(name="Marie Curie", type="Person"),
        Node(name="Pierre Curie", type="Person"),
        Node(name="Nobel Prize", type="Award"),
        Node(name="University of Paris", type="Organization"),
        Node(name="Polish", type="Nationality", properties={"European": True}),
        Node(name="French", type="Nationality", properties={"European": True}),
        Node(name="Physicist", type="Profession"),
        Node(name="Chemist", type="Profession"),
        Node(name="Radioactivity", type="Scientific concept"),
        Node(name="Professor", type="Profession"),
    ]
    expected_edges = {
        Relation(Node("Marie Curie", "Person"), Node("Polish", "Nationality"), "HAS_NATIONALITY"),
        Relation(Node("Marie Curie", "Person"), Node("French", "Nationality"), "HAS_NATIONALITY"),
        Relation(
            Node("Marie Curie", "Person"), Node("Physicist", "Profession"), "HAS_PROFESSION"
        ),
        Relation(Node("Marie Curie", "Person"), Node("Chemist", "Profession"), "HAS_PROFESSION"),
        Relation(
            Node("Marie Curie", "Person"), Node("Professor", "Profession"), "HAS_PROFESSION"
        ),
        Relation(
            Node("Marie Curie", "Person"),
            Node("Radioactivity", "Scientific concept"),
            "RESEARCHED",
        ),
        Relation(Node("Marie Curie", "Person"), Node("Nobel Prize", "Award"), "WON"),
        Relation(Node("Marie Curie", "Person"), Node("Pierre Curie", "Person"), "MARRIED_TO"),
        Relation(
            Node("Marie Curie", "Person"),
            Node("University of Paris", "Organization"),
            "WORKED_AT",
        ),
    }
    assert_that(result_edges, contains_exactly(*expected_edges))
    assert_that(result_nodes, contains_exactly(*expected_nodes))


def test_fuzzy_search(marie_curie: DataFixture) -> None:
    if not marie_curie.has_embeddings:
        pytest.skip("Fuzzy search requires embeddings. Run with openai environment variables")
    result_nodes = marie_curie.graph_store.graph.query_nearest_nodes(["Marie", "Poland"])
    expected_nodes = [
        Node(name="Marie Curie", type="Person"),
        Node(name="Polish", type="Nationality", properties={"European": True}),
    ]
    assert_that(result_nodes, contains_exactly(*expected_nodes))
