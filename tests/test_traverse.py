from precisely import assert_that, contains_exactly

from knowledge_graph.traverse import Relation, traverse

from .conftest import DataFixture


def test_traverse_marie_curie(marie_curie: DataFixture) -> None:
    results = traverse(
        "Marie Curie",
        steps=1,
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    expected = {
        Relation("Marie Curie", "Polish", "HAS_NATIONALITY"),
        Relation("Marie Curie", "French", "HAS_NATIONALITY"),
        Relation("Marie Curie", "Physicist", "HAS_PROFESSION"),
        Relation("Marie Curie", "Chemist", "HAS_PROFESSION"),
        Relation("Marie Curie", "Professor", "HAS_PROFESSION"),
        Relation("Marie Curie", "Radioactivity", "RESEARCHED"),
        Relation("Marie Curie", "Nobel Prize", "WON"),
        Relation("Marie Curie", "Pierre Curie", "MARRIED_TO"),
        Relation("Marie Curie", "University of Paris", "WORKED_AT"),
    }
    assert_that(results, contains_exactly(*expected))

    results = traverse(
        "Marie Curie",
        steps=2,
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    expected.add(Relation("Pierre Curie", "Nobel Prize", "WON"))
    assert_that(results, contains_exactly(*expected))


async def test_atraverse_marie_curie(marie_curie: DataFixture) -> None:
    results = traverse(
        "Marie Curie",
        steps=1,
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    expected = {
        Relation("Marie Curie", "Polish", "HAS_NATIONALITY"),
        Relation("Marie Curie", "French", "HAS_NATIONALITY"),
        Relation("Marie Curie", "Physicist", "HAS_PROFESSION"),
        Relation("Marie Curie", "Chemist", "HAS_PROFESSION"),
        Relation("Marie Curie", "Professor", "HAS_PROFESSION"),
        Relation("Marie Curie", "Radioactivity", "RESEARCHED"),
        Relation("Marie Curie", "Nobel Prize", "WON"),
        Relation("Marie Curie", "Pierre Curie", "MARRIED_TO"),
        Relation("Marie Curie", "University of Paris", "WORKED_AT"),
    }
    assert_that(results, contains_exactly(*expected))

    results = traverse(
        "Marie Curie",
        steps=2,
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    expected.add(Relation("Pierre Curie", "Nobel Prize", "WON"))
    assert_that(results, contains_exactly(*expected))