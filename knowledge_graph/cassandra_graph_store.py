from typing import Any, Dict, List, Optional, Tuple, Iterator
from langchain_community.graphs.graph_store import GraphStore
from langchain_community.graphs.graph_document import GraphDocument

from langchain_core.runnables import Runnable, RunnableLambda

from cassandra.query import BatchStatement, PreparedStatement
import cassio
from cassio.config import check_resolve_session, check_resolve_keyspace

from .traverse import atraverse, traverse
from .utils import batched


class CassandraGraphStore(GraphStore):
    def __init__(self,
                 node_table: str = "entities",
                 edge_table: str = "relationships",
                 keyspace: Optional[str] = None) -> None:
        cassio.init(auto=True)
        self._session = check_resolve_session()
        keyspace = keyspace or check_resolve_keyspace()
        self._keyspace = keyspace

        self._node_table = node_table
        self._edge_table = edge_table

        self._session.set_keyspace(keyspace)
        self._session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {node_table} (
                id TEXT,
                type TEXT,
                PRIMARY KEY (id, type)
            );
            """
        )

        self._session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {edge_table} (
                source TEXT,
                target TEXT,
                type TEXT,
                PRIMARY KEY (source, target, type)
            );
            """
        )

        self._insert_node = self._session.prepare(
            f"INSERT INTO {node_table} (id, type) VALUES (?, ?)"
        )

        self._insert_relationship = self._session.prepare(
            f"INSERT INTO {edge_table} (source, target, type) VALUES (?, ?, ?)"
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
                    (edge.source.id, edge.target.id, edge.type),
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
