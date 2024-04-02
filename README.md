# Example GraphRAG using Astra

This includes some code (CassandraGraphStore) which could be added to LangChain or RAGStack to write LangChain's `GraphDocuments` to Cassandra tables. It also includes code to create a runnable for retrieving knowledge triples from Cassandra.

The notebook shows this working on an example snippet from LangChain's docs.

To run, copy `env.template` to `.env`.