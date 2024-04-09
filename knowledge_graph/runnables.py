from typing import List, Optional
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.graph_transformers.llm import optional_enum_field

QUERY_KEYWORD_EXTRACT_PROMPT = (
    "A question is provided below. Given the question, extract up to 5 "
    "keywords from the text. Focus on extracting the keywords that we can use "
    "to best lookup answers to the question. Avoid stopwords.\n"
    "---------------------\n"
    "{question}\n"
    "---------------------\n"
    "{format_instructions}\n"
)

def extract_entities(
        llm: BaseChatModel,
        keyword_extraction_prompt: str = QUERY_KEYWORD_EXTRACT_PROMPT,
        node_types: Optional[List[str]] = None,
) -> Runnable:
    """
    Return a keyword-extraction runnable.

    This will expect a dictionary containing the `"question"` to extract keywords from.

    Parameters:
    - llm: The LLM to use for extracting entities.
    - node_types: List of node types to extract.
    - keyword_extraction_prompt: The prompt to use for requesting entities.
      This should include the `{question}` being asked as well as the `{format_instructions}`
      which describe how to produce the output.
    """
    prompt = ChatPromptTemplate.from_messages(keyword_extraction_prompt)
    assert "question" in prompt.input_variables
    assert "format_instructions" in prompt.input_variables

    class SimpleNode(BaseModel):
        """Represents a node in a graph with associated properties."""

        id: str = Field(description="Name or human-readable unique identifier.")
        type: str = optional_enum_field(
            node_types, description="The type or label of the node."
        )

    output_parser = JsonOutputParser(pydantic_object=SimpleNode)
    return (
        RunnablePassthrough.assign({
            "format_instructions": output_parser.get_format_instructions(),
        })
        | ChatPromptTemplate.from_messages([keyword_extraction_prompt])
        | llm
        | output_parser
    )