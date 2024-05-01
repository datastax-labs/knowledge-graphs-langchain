from os import path
from typing import Dict, List, Sequence, Union, cast

from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_experimental.graph_transformers.llm import (
    _Graph,
    create_simple_model,
    map_to_base_node,
    map_to_base_relationship,
)

from knowledge_graph.traverse import Node, Relation


class NodeSchema(BaseModel):
    type: str
    """The name of the node type."""

    description: str
    """Description of the node type."""


class EdgeSchema(BaseModel):
    type: str
    """The name of the edge type."""

    description: str
    """Description of the edge type."""


class RelationshipSchema(BaseModel):
    edge_type: str
    """The name of the edge type for the relationhsip."""

    source_types: List[str]
    """The node types for the source of the relationship."""

    target_types: List[str]
    """The node types for the target of the relationship."""

    description: str
    """Description of the relationship."""


class Example(BaseModel):
    input: str
    """The source input."""

    nodes: Sequence[Node]
    """The extracted example nodes."""

    edges: Sequence[Relation]
    """The extracted example relationhsips."""


class KnowledgeSchema(BaseModel):
    nodes: List[NodeSchema]
    """Allowed node types for the knowledge schema."""

    relationships: List[RelationshipSchema]
    """Allowed relationships for the knowledge schema."""

    examples: List[Example] = []
    """Example extractions."""


class KnowledgeSchemaValidator:
    def __init__(self, schema: KnowledgeSchema) -> None:
        self._schema = schema

        self._nodes = {node.type: node for node in schema.nodes}

        self._relationships: Dict[str, List[RelationshipSchema]] = {}
        for r in schema.relationships:
            self._relationships.setdefault(r.edge_type, []).append(r)

            # TODO: Validate the relationship.
            # source/target type should exist in nodes, edge_type should exist in edges

    def validate_graph_document(self, document: GraphDocument):
        e = ValueError("Invalid graph document for schema")
        for node_type in {node.type for node in document.nodes}:
            if node_type not in self._nodes:
                e.add_note(f"No node type '{node_type}")
        for r in document.relationships:
            relationships = self._relationships.get(r.edge_type, None)
            if relationships is None:
                e.add_note(f"No edge type '{r.edge_type}")
            else:
                relationship = next(
                    (
                        candidate
                        for candidate in relationships
                        if r.source_type in candidate.source_types
                        if r.target_type in candidate.target_types
                    )
                )
                if relationship is None:
                    e.add_note(
                        f"No relationship allows ({r.source_id} -> {r.type} -> {r.target.type})"
                    )

        if e.__notes__:
            raise e


TEMPLATE_PATH = path.join(path.dirname(__file__), "prompt_templates")


def _format_example(idx: int, example: Example) -> str:
    lines = [
        f"Example {idx} Input: {example.input}\n" f"Example {idx} Nodes: ",
    ]

    def f_node(node: Node) -> str:
        return f"{{ name: {node.name}, type: {node.type} }}"

    lines.extend([f"- {f_node(node)}" for node in example.nodes])
    lines.append(f"Example {idx} Edges: ")
    lines.extend(
        [
            f"- {{ source: {f_node(e.source)}, target: {f_node(e.target)}, type: {e.type} }}"
            for e in example.edges
        ]
    )
    return "\n".join(lines)


def _extraction_prompt(schema: KnowledgeSchema) -> SystemMessagePromptTemplate:
    def fmt_node(node: NodeSchema) -> str:
        return f"- Node type {node.type}: {node.description}"

    def fmt_relationship(rel: RelationshipSchema) -> str:
        return (
            f"- Edge type {rel.edge_type}: {rel.description}\n"
            f"  Source node types: {rel.source_types}\n"
            f"  Target node types: {rel.target_types}\n"
        )

    node_types = "\n".join(map(fmt_node, schema.nodes))
    relationship_patterns = "\n".join(map(fmt_relationship, schema.relationships))

    return SystemMessagePromptTemplate(
        prompt=PromptTemplate.from_file(path.join(TEMPLATE_PATH, "extraction.md")).partial(
            node_types=node_types, relationship_patterns=relationship_patterns
        )
    )


class KnowledgeSchemaExtractor:
    def __init__(self, llm: BaseChatModel, schema: KnowledgeSchema, strict: bool = False) -> None:
        self._validator = KnowledgeSchemaValidator(schema)
        self.strict = strict

        messages = [_extraction_prompt(schema)]

        if schema.examples:
            formatted = "\n\n".join(map(_format_example, schema.examples))
            messages.append(SystemMessagePromptTemplate(prompt=formatted))

        messages.append(HumanMessagePromptTemplate.from_template("Input: {input}"))

        prompt = ChatPromptTemplate.from_messages(messages)
        schema = create_simple_model(
            node_labels=[node.type for node in schema.nodes],
            rel_types=list({r.edge_type for r in schema.relationships}),
        )
        structured_llm = llm.with_structured_output(schema)
        self._chain = prompt | structured_llm

    def _process_response(
        self, document: Document, response: Union[Dict, BaseModel]
    ) -> GraphDocument:
        raw_graph = cast(_Graph, response)
        nodes = [map_to_base_node(node) for node in raw_graph.nodes] if raw_graph.nodes else []
        relationships = (
            [map_to_base_relationship(rel) for rel in raw_graph.relationships]
            if raw_graph.relationships
            else []
        )

        document = GraphDocument(nodes=nodes, relationships=relationships, source=document)

        if self.strict:
            self._validator.validate_graph_document(document)

        return document

    def extract(self, documents: List[Document]) -> List[GraphDocument]:
        # TODO: Define an async version of extraction?
        responses = self._chain.batch_as_completed(
            [{"input": doc.page_content} for doc in documents]
        )
        return [self._process_response(documents[idx], response) for idx, response in responses]
