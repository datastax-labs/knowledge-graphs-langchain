from typing import Any, Dict, Iterator, List, Optional, Tuple

from cassandra.cluster import Session
from cassandra.query import BatchStatement, PreparedStatement
from cassio.config import check_resolve_keyspace, check_resolve_session
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore
from langchain_core.runnables import Runnable, RunnableLambda

from .traverse import atraverse, traverse
from .utils import batched


class CassandraGraphStore(GraphStore):
    def __init__(
        self,
        node_table: str = "entities",
        edge_table: str = "relationships",
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
    ) -> None:
        """
        Create a Cassandra Graph Store.

        Before calling this, you must initialize cassio with `cassio.init`, or
        provide valid session and keyspace values.
        """
        session = check_resolve_session(session)
        keyspace = check_resolve_keyspace(keyspace)

        self._session = session
        self._keyspace = keyspace

        self._node_table = node_table
        self._edge_table = edge_table

        # Partition by `name` and cluster by `type`.
        # Each `(name, type)` pair is a unique node.
        # We can enumerate all `type` values for a given `name` to identify ambiguous terms.
        self._session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {keyspace}.{node_table} (
                name TEXT,
                type TEXT,
                PRIMARY KEY (name, type)
            );
            """
        )

        self._session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {keyspace}.{edge_table} (
                source_name TEXT,
                source_type TEXT,
                target_name TEXT,
                target_type TEXT,
                edge_type TEXT,
                PRIMARY KEY ((source_name, source_type), target_name, target_type, edge_type)
            );
            """
        )

        self._session.execute(
            f"""
            CREATE CUSTOM INDEX {edge_table}_type_index
            ON {keyspace}.{edge_table} (edge_type)
            USING 'StorageAttachedIndex';
            """
        )

        self._insert_node = self._session.prepare(
            f"INSERT INTO {keyspace}.{node_table} (name, type) VALUES (?, ?)"
        )

        self._insert_relationship = self._session.prepare(
            f"""
            INSERT INTO {keyspace}.{edge_table} (
                source_name, source_type, target_name, target_type, edge_type
            ) VALUES (?, ?, ?, ?, ?)
            """
        )

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        insertions = self._insertions(graph_documents, include_source)

        for batch in batched(insertions, n=50):
            batch_statement = BatchStatement()
            for statement, args in batch:
                batch_statement.add(statement, args)
            self._session.execute(batch_statement)

    def _insertions(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> Iterator[Tuple[PreparedStatement, Any]]:
        for graph_document in graph_documents:
            # TODO: if `include_source = True`, include entry connecting the
            # nodes we add to the `graph_document.source`.
            for node in graph_document.nodes:
                yield (self._insert_node, (node.id, node.type))
            for edge in graph_document.relationships:
                yield (
                    self._insert_relationship,
                    (edge.source.id, edge.source.type, edge.target.id, edge.target.type, edge.type),
                )

    # TODO: should this include the types of each node?
    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        steps = params.get("steps", 3)

        return self.as_runnable(steps=steps).invoke(query)

    def as_runnable(self, steps: int = 3) -> Runnable:
        """
        Return a retriever that queries this graph store for the entity or entities specified.

        Parameters:
        - steps: The maximum distance to follow from the starting points.
        """
        return RunnableLambda(func=traverse, afunc=atraverse).bind(
            edge_table=self._edge_table,
            steps=steps,
            session=self._session,
            keyspace=self._keyspace,
        )
