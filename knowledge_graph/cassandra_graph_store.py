from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterator
from langchain_community.graphs.graph_store import GraphStore
from langchain_community.graphs.graph_document import GraphDocument

from langchain_core.runnables import Runnable, RunnableLambda

from cassandra.cluster import Session
from cassandra.query import BatchStatement, PreparedStatement
import cassio
from cassio.config import check_resolve_session, check_resolve_keyspace

try:
    # Try importing the function from itertools (Python 3.12+)
    from itertools import batched
except ImportError:
    from itertools import islice
    from typing import Iterable, TypeVar

    # Fallback implementation for older Python versions

    T = TypeVar("T")

    # This is equivalent to `itertools.batched`, but that is only available in 3.12
    def batched(iterable: Iterable[T], n: int) -> Iterator[Iterator[T]]:
        if n < 1:
            raise ValueError('n must be at least one')
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch

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
        for nodes in batched(pending, 50):
            nodes = ", ".join([f"'{node}'" for node in nodes])
            query = f"SELECT source, target, type FROM relationships WHERE source IN ({nodes})"

            relations = session.execute(query)

            # Can't do this yet -- the `nodes` list isn't valid in this context (only valid in vector or list).
            # relations = session.execute("SELECT source, target, type FROM relationships WHERE source IN (%(nodes)s)", {"nodes": nodes})

            for relation in relations:
                results.add(relation)
                discovered.add(relation.target)

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
                PRIMARY KEY (source, target, type)
            );
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
        insertions = self._insertions(graph_documents, include_source)

        for batch in batched(insertions, n=50):
            batch_statement = BatchStatement()
            for (statement, args) in batch:
                batch_statement.add(statement, args)
            self._session.execute(batch_statement)

    def _insertions(self, graph_documents: List[GraphDocument], include_source: bool = False
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