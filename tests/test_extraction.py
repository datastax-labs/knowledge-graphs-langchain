from precisely import assert_that, contains_exactly
import pytest
from knowledge_graph.extraction import KnowledgeSchema, KnowledgeSchemaExtractor, NodeSchema, RelationshipSchema
from langchain_core.language_models import BaseChatModel
from langchain_community.graphs.graph_document import Node, Relationship
from langchain_core.documents import Document

TEST_KNOWLEDGE_SCHEMA = KnowledgeSchema(
    nodes = [
        NodeSchema(type = "Institution",
                   description="An institution, such as a business or university."),
        NodeSchema(type = "Award",
                   description="An award, such as the Nobel Prize or an Oscar."),
        NodeSchema(type = "Person",
                   description = "A person."),
        NodeSchema(type = "Discipline",
                   description = "An area of study, such as Biology or Chemistry."),
        NodeSchema(type = "Nationality",
                    description = "A nationality associated with people of a given country.")
    ],
    relationships = [
        RelationshipSchema(
            edge_type = "STUDIED",
            source_types = ["Person"],
            target_types = ["Discipline"],
            description = "The source person studied the target discipline.",
        ),
        RelationshipSchema(
            edge_type = "STUDIED_AT",
            source_types = ["Person"],
            target_types = ["Institution"],
            description = "The source person studied at the target institution.",
        ),
        RelationshipSchema(
            edge_type = "WORKED_AT",
            source_types = ["Person"],
            target_types = ["Institution"],
            description = "The source person worked at the target institution.",
        ),
        RelationshipSchema(
            edge_type = "RECEIVED",
            source_types = ["Person"],
            target_types = ["Award"],
            description = "The source person received the target award.",
        ),
        RelationshipSchema(
            edge_type = "HAS_NATIONALITY",
            source_types = ["Person"],
            target_types = ["Nationality"],
            description = "The source person has the target nationality.",
        ),
        RelationshipSchema(
            edge_type = "MARRIED_TO",
            source_types = ["Person"],
            target_types = ["Person"],
            description = "The source is married to the target. Marriage is symmetric so the reverse relationship should also exist."
        )
    ],
)

@pytest.fixture(scope="session")
def extractor(llm: BaseChatModel) -> KnowledgeSchemaExtractor:
    return KnowledgeSchemaExtractor(
        llm = llm,
        schema = TEST_KNOWLEDGE_SCHEMA,
    )


def test_extraction(extractor: KnowledgeSchemaExtractor):
    results = extractor.extract([Document(page_content="\n".join([
        "Marie Curie, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.",
        "She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.",
        "Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.",
        "She was, in 1906, the first woman to become a professor at the University of Paris."
    ]))])

    marie_curie = Node(id = "Marie Curie", type = "Person")
    polish = Node(id = "Polish", type = "Nationality")
    french = Node(id = "French", type = "Nationality")
    physicist = Node(id = "Physicist", type = "Discipline")
    chemist = Node(id = "Chemist", type = "Discipline")
    nobel_prize = Node(id = "Nobel Prize", type = "Award")
    pierre_curie = Node(id = "Pierre Curie", type = "Person")

    # Annoyingly, the LLM seems to upper-case `of`. We probably need some instructions around
    # putting things into standard title case, etc.
    university_of_paris = Node(id = "University Of Paris", type = "Institution")

    assert_that(
        results[0].nodes,
        contains_exactly(
            marie_curie,
            polish,
            french,
            physicist,
            chemist,
            nobel_prize,
            pierre_curie,
            university_of_paris,
        )
    )
    assert_that(
        results[0].relationships,
        contains_exactly(
            Relationship(source = marie_curie, target=polish, type="HAS_NATIONALITY"),
            Relationship(source = marie_curie, target=french, type="HAS_NATIONALITY"),
            Relationship(source = marie_curie, target=physicist, type="STUDIED"),
            Relationship(source = marie_curie, target=chemist, type="STUDIED"),
            Relationship(source = marie_curie, target=nobel_prize, type="RECEIVED"),
            Relationship(source = pierre_curie, target=nobel_prize, type="RECEIVED"),
            Relationship(source = marie_curie, target=university_of_paris, type="WORKED_AT"),
            Relationship(source = marie_curie, target=pierre_curie, type="MARRIED_TO"),
            Relationship(source = pierre_curie, target=marie_curie, type="MARRIED_TO"),
        )
    )