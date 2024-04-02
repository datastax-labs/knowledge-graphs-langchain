from typing import Any, Dict, List, Optional, Sequence
from langchain_community.graphs.graph_store import GraphStore
from langchain_community.graphs.graph_document import GraphDocument

from langchain_core.runnables import Runnable, RunnableLambda

from cassandra.cluster import Session
from cassandra.query import BatchStatement
import cassio
from cassio.config import check_resolve_session, check_resolve_keyspace

QUERY_RELATIONSHIPS: str = """
    SELECT source, target, type
    FROM relationships
    WHERE source = %(node)s
"""

def graph_retriever(query: str | Sequence[str],
                    session: Session,
                    steps: int = 3) -> Sequence[str]:
    # TODO: Entity extraction from query
    visited = set()
    pending = set()

    if isinstance(query, str):
        pending.update([query])
    else:
        pending.update(query)

    results = set()

    for _ in range(steps):
        if not pending:
            break

        discovered = set()
        # TODO: Make async?
        for node in pending:
            relations = session.execute(
                QUERY_RELATIONSHIPS, {"node": node}
            )
            relations = list(relations)

            results.update(relations)

            discovered.update(map(lambda r: r.target, relations))

        visited.update(pending)
        pending = discovered.difference(visited)

    return [f"{r.source} -> {r.target}: {r.type}" for r in results]

class CassandraGraphStore(GraphStore):
    def __init__(self, keyspace: Optional[str] = None) -> None:
        cassio.init(auto=True)
        self._session = check_resolve_session()
        keyspace = keyspace or check_resolve_keyspace()

        self._session.set_keyspace(keyspace)
        self._session.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT,
                type TEXT,
                PRIMARY KEY (id, type)
            );
            """
        )

        self._session.execute(
            """
            CREATE TABLE IF NOT EXISTS relationships (
                source TEXT,
                target TEXT,
                type TEXT,
                PRIMARY KEY ((source, target), type)
            );
            """
        )

        self._session.execute(
            """
            CREATE CUSTOM INDEX IF NOT EXISTS relationships_source_idx
            ON relationships (source)
            USING 'StorageAttachedIndex';
            """
        )

        self._insert_node = self._session.prepare(
            "INSERT INTO entities (id, type) VALUES (?, ?)"
        )

        self._insert_relationship = self._session.prepare(
            "INSERT INTO relationships (source, target, type) VALUES (?, ?, ?)"
        )

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        batch = BatchStatement()
        for graph_document in graph_documents:
            # TODO: if `include_source = True`, include entry connecting the
            # nodes we add to the `graph_document.source`.
            for node in graph_document.nodes:
                batch.add(self._insert_node, (node.id, node.type))
            for edge in graph_document.relationships:
                batch.add(
                    self._insert_relationship,
                    (edge.source.id, edge.target.id, edge.type),
                )

        # TODO: Do we need to roll the batch if it gets too large?
        self._session.execute(batch)

    # TODO: should this include the types of each node?
    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        steps = params.get("steps", 3)

        return self.as_runnable(steps = steps).invoke(query)

    def as_runnable(
        self, steps: int = 3
    ) -> Runnable:
        """
        Return a retriever that queries this graph store.

        Parameters:
        - llm: The language model to use for extracting entities from the query.
        - steps: The maximum distance to follow from the starting points.
        """
        return RunnableLambda(graph_retriever).bind(
            session = self._session,
            steps = steps
        )