from precisely import assert_that, contains_exactly
import pytest

from knowledge_graph.traverse import Relation, atraverse, traverse

from .conftest import DataFixture


def test_traverse_marie_curie(marie_curie: DataFixture) -> None:
    results = traverse(
        start=("Marie Curie", "Person"),
        steps=1,
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    expected = {
        Relation("Marie Curie", "Person", "Polish", "Nationality", "HAS_NATIONALITY"),
        Relation("Marie Curie", "Person", "French", "Nationality", "HAS_NATIONALITY"),
        Relation("Marie Curie", "Person", "Physicist", "Profession", "HAS_PROFESSION"),
        Relation("Marie Curie", "Person", "Chemist", "Profession", "HAS_PROFESSION"),
        Relation("Marie Curie", "Person", "Professor", "Profession", "HAS_PROFESSION"),
        Relation("Marie Curie", "Person", "Radioactivity", "Scientific concept", "RESEARCHED"),
        Relation("Marie Curie", "Person", "Nobel Prize", "Award", "WON"),
        Relation("Marie Curie", "Person", "Pierre Curie", "Person", "MARRIED_TO"),
        Relation("Marie Curie", "Person", "University of Paris", "Organization", "WORKED_AT"),
    }
    assert_that(results, contains_exactly(*expected))

    results = traverse(
        start=("Marie Curie", "Person"),
        steps=2,
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    expected.add(Relation("Pierre Curie", "Person", "Nobel Prize", "Award", "WON"))
    assert_that(results, contains_exactly(*expected))


async def test_atraverse_marie_curie(marie_curie: DataFixture) -> None:
    results = await atraverse(
        start=("Marie Curie", "Person"),
        steps=1,
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    expected = {
        Relation("Marie Curie", "Person", "Polish", "Nationality", "HAS_NATIONALITY"),
        Relation("Marie Curie", "Person", "French", "Nationality", "HAS_NATIONALITY"),
        Relation("Marie Curie", "Person", "Physicist", "Profession", "HAS_PROFESSION"),
        Relation("Marie Curie", "Person", "Chemist", "Profession", "HAS_PROFESSION"),
        Relation("Marie Curie", "Person", "Professor", "Profession", "HAS_PROFESSION"),
        Relation("Marie Curie", "Person", "Radioactivity", "Scientific concept", "RESEARCHED"),
        Relation("Marie Curie", "Person", "Nobel Prize", "Award", "WON"),
        Relation("Marie Curie", "Person", "Pierre Curie", "Person", "MARRIED_TO"),
        Relation("Marie Curie", "Person", "University of Paris", "Organization", "WORKED_AT"),
    }
    assert_that(results, contains_exactly(*expected))

    results = await atraverse(
        start=("Marie Curie", "Person"),
        steps=2,
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    expected.add(Relation("Pierre Curie", "Person", "Nobel Prize", "Award", "WON"))
    assert_that(results, contains_exactly(*expected))


def test_traverse_marie_curie_filtered_edges(marie_curie: DataFixture) -> None:
    results = traverse(
        start=("Marie Curie", "Person"),
        steps=1,
        edge_filters=["edge_type = 'HAS_NATIONALITY'"],
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    expected = {
        Relation("Marie Curie", "Person", "Polish", "Nationality", "HAS_NATIONALITY"),
        Relation("Marie Curie", "Person", "French", "Nationality", "HAS_NATIONALITY"),
    }
    assert_that(results, contains_exactly(*expected))


async def test_atraverse_marie_curie_filtered_edges(marie_curie: DataFixture) -> None:
    results = await atraverse(
        start=("Marie Curie", "Person"),
        steps=1,
        edge_filters=["edge_type = 'HAS_NATIONALITY'"],
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    expected = {
        Relation("Marie Curie", "Person", "Polish", "Nationality", "HAS_NATIONALITY"),
        Relation("Marie Curie", "Person", "French", "Nationality", "HAS_NATIONALITY"),
    }
    assert_that(results, contains_exactly(*expected))
