from os import path
from typing import Dict, List, Union, cast

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

from knowledge_graph.knowledge_schema import (
    Example,
    KnowledgeSchema,
    KnowledgeSchemaValidator,
    NodeSchema,
    RelationshipSchema,
)
from knowledge_graph.traverse import Node

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
