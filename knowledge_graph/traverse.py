import asyncio
from typing import Iterable, NamedTuple, Optional, Sequence, Set, Tuple

from cassandra.cluster import ResponseFuture, Session
from cassandra.query import named_tuple_factory
from cassio.config import check_resolve_keyspace, check_resolve_session

from .utils import batched


class Relation(NamedTuple):
    source_name: str
    source_type: str
    target_name: str
    target_type: str
    type: str

    def __repr__(self):
        return f"{self.source_name}({self.source_type}) -> {self.target_name}({self.target_type}): {self.type}"


def _parse_relation(row) -> Relation:
    return Relation(
        source_name=row.source_name,
        source_type=row.source_type,
        target_name=row.target_name,
        target_type=row.target_type,
        type=row.type,
    )


def _edge_query(
    edge_table: str,
    edge_source_name: str,
    edge_source_type: str,
    edge_target_name: str,
    edge_target_type: str,
    edge_type: str,
    predicates: Sequence[str],
) -> str:
    """Return the query for the edges from a given source."""
    query = f"""
        SELECT
            {edge_source_name} AS source_name,
            {edge_source_type} AS source_type,
            {edge_target_name} AS target_name,
            {edge_target_type} AS target_type,
            {edge_type} AS type
        FROM {edge_table}
        WHERE {edge_source_name} = ?
        AND {edge_source_type} = ?"""
    if predicates:
        query = "\n        AND ".join([query] + predicates)
    return query

def traverse(
    start: Tuple[str, str] | Sequence[Tuple[str, str]],
    edge_table: str,
    edge_source_name: str = "source_name",
    edge_source_type: str = "source_type",
    edge_target_name: str = "target_name",
    edge_target_type: str = "target_type",
    edge_type: str = "edge_type",
    edge_filters: Sequence[str] = (),
    steps: int = 3,
    session: Optional[Session] = None,
    keyspace: Optional[str] = None,
) -> Iterable[Relation]:
    """
    Traverse the graph from the given starting nodes and return the resulting sub-graph.

    Parameters:
    - start: The starting node or nodes.
    - edge_table: The table containing the edges.
    - edge_source_name: The name of the column containing edge source names.
    - edge_source_type: The name of the column containing edge source types.
    - edge_target_name: The name of the column containing edge target names.
    - edge_target_type: The name of the column containing edge target types.
    - edge_type: The name of the column containing edge types.
    - edge_filters: Filters to apply to the edges being traversed.
    - steps: The number of steps of edges to follow from a start node.
    - session: The session to use for executing the query. If not specified,
      it will use th default cassio session.
    - keyspace: The keyspace to use for the query. If not specified, it will
      use the default cassio keyspace.

    Returns:
    An iterable over relations in the traversed sub-graph.
    """
    return asyncio.get_event_loop().run_until_complete(atraverse(
        start=start,
        edge_table=edge_table,
        edge_source_name=edge_source_name,
        edge_source_type=edge_source_type,
        edge_target_name=edge_target_name,
        edge_target_type=edge_target_type,
        edge_type=edge_type,
        edge_filters=edge_filters,
        steps=steps,
        session=session,
        keyspace=keyspace,
    ))


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
        page = [_parse_relation(r) for r in await self.current_page_future]

        if self.response_future.has_more_pages:
            self.current_page_future = asyncio.Future()
            self.response_future.start_fetching_next_page()
            return (self.depth, page, self)
        else:
            return (self.depth, page, None)


async def atraverse(
    start: Tuple[str, str] | Sequence[Tuple[str, str]],
    edge_table: str,
    edge_source_name: str = "source_name",
    edge_source_type: str = "source_type",
    edge_target_name: str = "target_name",
    edge_target_type: str = "target_type",
    edge_type: str = "edge_type",
    edge_filters: Sequence[str] = [],
    steps: int = 3,
    session: Optional[Session] = None,
    keyspace: Optional[str] = None,
) -> Iterable[Relation]:
    """
    Async traversal of the graph from the given starting nodes and return the resulting sub-graph.

    Parameters:
    - start: The starting node or nodes.
    - edge_table: The table containing the edges.
    - edge_source_name: The name of the column containing edge source names.
    - edge_source_type: The name of the column containing edge source types.
    - edge_target_name: The name of the column containing edge target names.
    - edge_target_type: The name of the column containing edge target types.
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
    An iterable over relations in the traversed sub-graph.
    """

    session = check_resolve_session(session)
    keyspace = check_resolve_keyspace(keyspace)

    # Prepare the query.
    #
    # We reprepare this for each traversal since each call may have different
    # filters.
    #
    # TODO: We should cache this at least for the common case of no-filters.
    query = session.prepare(
        _edge_query(
            edge_table=edge_table,
            edge_source_name=edge_source_name,
            edge_source_type=edge_source_type,
            edge_target_name=edge_target_name,
            edge_target_type=edge_target_type,
            edge_type=edge_type,
            predicates=edge_filters,
        ),
        keyspace=keyspace
    )

    def fetch_relation(
            tg: asyncio.TaskGroup,
            depth: int,
            source_name: str,
            source_type: str) -> AsyncPagedQuery:
        paged_query = AsyncPagedQuery(depth, session.execute_async(query, (source_name, source_type)))
        return tg.create_task(paged_query.next())

    results = set()
    async with asyncio.TaskGroup() as tg:
        if isinstance(start, tuple):
            start = [start]

        discovered = {t: 0 for t in start}
        pending = {
            fetch_relation(tg, 1, source_name, source_type)
            for source_name, source_type in start
        }

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
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
                        target = (r.target_name, r.target_type)
                        previous = discovered.get(target, steps + 1)
                        if depth < previous:
                            discovered[target] = depth
                            to_visit.add(target)

                    for target_name, target_type in to_visit:
                        pending.add(fetch_relation(tg, depth + 1, target_name, target_type))

    return results
