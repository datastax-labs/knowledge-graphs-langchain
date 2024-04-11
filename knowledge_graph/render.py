import graphviz
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

def _node_name(node: Node) -> str:
    return f"{node.id} [{node.type}]"

def render_graph_document(doc: GraphDocument) -> graphviz.Digraph:
    dot = graphviz.Digraph()

    for n in doc.nodes:
        dot.node(_node_name(n))

    for r in doc.relationships:
        dot.edge(_node_name(r.source), _node_name(r.target), r.type)

    return dot
