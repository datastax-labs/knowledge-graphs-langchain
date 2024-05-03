from typing import Dict, Sequence, cast
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough

from knowledge_graph.knowledge_schema import KnowledgeSchema
from knowledge_graph.templates import load_template

class KnowledgeSchemaInferer():
    def __init__(self, llm: BaseChatModel) -> None:
        infer_prompt = load_template(
            "schema_inference.md",
        )
        # TODO: Use "full" output so we can detect parsing errors?
        structured_llm = llm.with_structured_output(KnowledgeSchema)

        infer_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(prompt = load_template("schema_inference.md")),
            HumanMessagePromptTemplate.from_template("Input: {input}")
        ])
        self._infer_chain = infer_prompt | structured_llm

        merge_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(prompt = load_template("merge_schema.md")),
            HumanMessagePromptTemplate.from_template("Schemas:\n{schemas}")
        ])
        def format_schemas(schemas: Sequence[str]) -> Dict:
            return { "schemas": "\n\n".join(schemas) }

        # TODO: Support fold or tree summarization
        self._merge_chain = RunnablePassthrough() | format_schemas | merge_prompt | structured_llm
        self._llm = llm

    def infer_schemas_from(self, documents: Sequence[Document]) -> Sequence[KnowledgeSchema]:
        responses = self._infer_chain.batch([{"input": doc.page_content} for doc in documents])
        return cast(Sequence[KnowledgeSchema], responses)

    def merge_schemas(self,
                      schemas: Sequence[KnowledgeSchema],
                      token_max: int = 1000000) -> KnowledgeSchema:
        merged = None
        to_merge = []

        # NOTE: We don't include the token size of the prompt. We assume that is removed from
        # the tokens the LLM can be provided (eg., token_max <= max_tokens - prompt_tokens).
        token_size = 0
        for schema in schemas:
            schema_str = f"Schema {len(to_merge)}:\n```yaml\n{schema.to_yaml_str()}\n```"
            schema_tokens = self._llm.get_num_tokens(schema_str)

            if token_size + schema_tokens > token_max:
                print(f"Merging: {to_merge}")
                merged = self._merge_chain(to_merge)
                merged_str = f"Schema 0:\n```yaml\n{schema.to_yaml_str()}\n```"
                to_merge = [merged_str]
                token_size = self._llm.get_num_tokens(merged_str)
            else:
                if len(to_merge) == 0:
                    merged = schema
                to_merge.append(schema_str)
                token_size += schema_tokens

        if merged:
            return merged
        else:
            print(f"Merging: {to_merge}")
            return self._merge_chain(to_merge)