import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterator, NamedTuple
from langchain_community.graphs.graph_store import GraphStore
from langchain_community.graphs.graph_document import GraphDocument

from langchain_core.runnables import Runnable, RunnableLambda

from cassandra.cluster import Session, ResponseFuture
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
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch


class Relation(NamedTuple):
    source: str
    target: str
    type: str

    def __repr__(self):
        return f"{self.source} -> {self.target}: {self.type}"


def _query_string(
    sources: Sequence[str],
    edge_table: str,
    edge_source: str,
    edge_target: str,
    edge_type: str,
    predicates: Sequence[str],
) -> str:
    """Return the query string for the given number of sources.

    Ideally, this would be something like `source IN %s` and we wouldn't need to
    expand it for each length, but currently that produces an error since it
    doesn't want to accept a list in that context. Instead, we need to unroll it
    to `source IN (%s, %s, ...)` for the number of arguments.
    """
    sources = ", ".join([f"'{n}'" for n in sources])
    lines = [
        f"SELECT {edge_source} AS source, {edge_target} AS target, {edge_type} AS type",
        f"FROM {edge_table}",
        f"WHERE {edge_source} IN ({sources})",
    ]
    if predicates:
        lines.extend(map(lambda predicate: f"AND {predicate}"))
    return " ".join(lines)


DEFAULT_MAX_ELEMENTS_PER_BATCH = 50


def traverse(
    start: str | Sequence[str],
    edge_table: str,
    edge_source: str = "source",
    edge_target: str = "target",
    edge_type: str = "type",
    edge_filters: Sequence[str] = (),
    steps: int = 3,
    session: Optional[Session] = None,
    keyspace: Optional[str] = None,
    max_elements_per_batch: int = DEFAULT_MAX_ELEMENTS_PER_BATCH,
) -> Iterable[Relation]:
    """
    Traverse the graph from the given starting nodes.

    Parameters:
    - start: The starting node or nodes.
    - edge_table: The table containing the edges.
    - edge_source: The name of the column containing edge sources.
    - edge_target: The name of the column containing edge targets.
    - edge_type: The name of the column containing edge types.
    - edge_filters: Filters to apply to the edges being traversed.
    - steps: The number of steps of edges to follow from a start node.
    - session: The session to use for executing the query. If not specified,
      it will use th default cassio session.
    - keyspace: The keyspace to use for the query. If not specified, it will
      use the default cassio keyspace.

    Returns:
    An iterable containing all relations visited by the traversal.
    """
    session = check_resolve_session(session)
    keyspace = check_resolve_keyspace(keyspace)

    visited = set()
    pending = set()

    if isinstance(start, str):
        pending.update([start])
    else:
        pending.update(start)

    results = set()

    for _ in range(steps):
        if not pending:
            break

        discovered = set()

        for nodes in batched(pending, max_elements_per_batch):
            query = _query_string(
                sources=nodes,
                edge_table=edge_table,
                edge_source=edge_source,
                edge_target=edge_target,
                edge_type=edge_type,
                predicates=edge_filters,
            )
            relations = session.execute(query)

            for relation in relations:
                results.add(
                    Relation(
                        source=relation.source,
                        target=relation.target,
                        type=relation.type,
                    )
                )
                discovered.add(relation.target)

        visited.update(pending)
        pending = discovered.difference(visited)

    return results


class AsyncPagedQuery(object):
    def __init__(self, depth: int, response_future: ResponseFuture):
        self.loop = asyncio.get_running_loop()
        self.depth = depth
        self.response_future = response_future
        self.current_page_future = asyncio.Future()
        self.response_future.add_callbacks(self._handle_page, self._handle_error)

    def _handle_page(self, rows):
        self.loop.call_soon_threadsafe(self.current_page_future.set_result, rows)

    def _handle_error(self, error):
        self.loop.call_soon_threadsafe(self.current_page_future.set_exception, error)

    async def next(self):
        page = await self.current_page_future

        if self.response_future.has_more_pages:
            self.current_page_future = asyncio.Future()
            self.response_future.start_fetching_next_page()
            return (self.depth, page, self)
        else:
            return (self.depth, page, None)


async def atraverse(
    start: str | Sequence[str],
    edge_table: str,
    edge_source: str = "source",
    edge_target: str = "target",
    edge_type: str = "type",
    edge_filters: Sequence[str] = [],
    steps: int = 3,
    session: Optional[Session] = None,
    keyspace: Optional[str] = None,
    max_elements_per_batch: int = DEFAULT_MAX_ELEMENTS_PER_BATCH,
) -> Iterable[Relation]:
    """
    Async traversal of the graph from the given starting nodes.

    Parameters:
    - start: The starting node or nodes.
    - edge_table: The table containing the edges.
    - edge_source: The name of the column containing edge sources.
    - edge_target: The name of the column containing edge targets.
    - edge_type: The name of the column containing edge types.
    - edge_filters: Filters to apply to the edges being traversed.
      Currently, this is specified as a dictionary containing the name
      of the edge field to filter on and the CQL predicate to apply.
      For example `{"foo": "IN ['a', 'b', 'c']"}`.
    - steps: The number of steps of edges to follow from a start node.
    - session: The session to use for executing the query. If not specified,
      it will use th default cassio session.
    - keyspace: The keyspace to use for the query. If not specified, it will
      use the default cassio keyspace.

    Returns:
    An iterable containing all relations visited by the traversal.
    """

    session = check_resolve_session(session)
    keyspace = check_resolve_keyspace(keyspace)

    def fetch_relations(depth: int, sources: Sequence[str]) -> AsyncPagedQuery:
        query = _query_string(
            sources=sources,
            edge_table=edge_table,
            edge_source=edge_source,
            edge_target=edge_target,
            edge_type=edge_type,
            predicates=edge_filters,
        )

        return AsyncPagedQuery(depth, session.execute_async(query))

    results = set()
    async with asyncio.TaskGroup() as tg:
        if isinstance(start, str):
            start = [start]

        discovered = {t: 0 for t in start}
        pending = [tg.create_task(fetch_relations(1, start).next())]

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for future in done:
                depth, relations, more = future.result()
                for relation in relations:
                    results.add(relation)

                # Schedule the future for more results from the same query.
                if more is not None:
                    pending.add(tg.create_task(more.next()))

                # Schedule futures for the next step.
                if depth < steps:
                    # We've found a path of length `depth` to each of the targets.
                    # We need to update `discovered` to include the shortest path.
                    # And build `to_visit` to be all of the targets for which this is
                    # the new shortest path.
                    to_visit = set()
                    for r in relations:
                        previous = discovered.get(r.target, steps + 1)
                        if depth < previous:
                            discovered[r.target] = depth
                            to_visit.add(r.target)

                    for target_batch in batched(to_visit, max_elements_per_batch):
                        pending.add(
                            tg.create_task(
                                fetch_relations(depth + 1, target_batch).next()
                            )
                        )

    return results


class CassandraGraphStore(GraphStore):
    def __init__(self, keyspace: Optional[str] = None) -> None:
        cassio.init(auto=True)
        self._session = check_resolve_session()
        keyspace = keyspace or check_resolve_keyspace()
        self._keyspace = keyspace

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
        Return a retriever that queries this graph store.

        Parameters:
        - llm: The language model to use for extracting entities from the query.
        - steps: The maximum distance to follow from the starting points.
        """
        return RunnableLambda(func=traverse, afunc=atraverse).bind(
            edge_table="relationships",
            steps=steps,
            session=self._session,
            keyspace=self._keyspace,
        )
