from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools.base import StructuredTool
from langgraph.prebuilt import create_react_agent
from rag_sql_agent.cfg import DEFAULT_CFG
from rag_sql_agent.utils import ConfigDict, BASE_DIR
from rag_sql_agent.utils.loaders import load_faiss_index
from rag_sql_agent.data_pipelines.create_vector_db import create_vector_db
from rag_sql_agent.data_pipelines.extract_entities import extract_entities_in_sql_db
import os


class LocalLLMAgent:
    def __init__(self, question, **kwargs):
        """Initialize the Local LLM Agent."""
        # Initialize the language model
        self.question = question
        self.cfg = ConfigDict({**DEFAULT_CFG, **kwargs})
        self.llm = ChatOpenAI(model=self.cfg.openai_model, temperature=0)
        # Load vector database
        self.embedding_model = OpenAIEmbeddings(model=self.cfg.openai_embeddings_model)
        # Determine the full path (absolute or relative)
        faiss_index_path = (
            self.cfg.faiss_index_name
            if os.path.isabs(self.cfg.faiss_index_name)
            else os.path.join(BASE_DIR, self.cfg.faiss_index_name)
        )
        # Check if the path exists, and create vector DB if it doesn't
        json_data = create_vector_db(self.embedding_model, self.cfg, faiss_index_path)

        # Load the FAISS vector store
        self.vector_store = load_faiss_index(
            self.embedding_model, faiss_index_path, allow_dangerous_deserialization=True
        )
        db_full_path = (
            self.cfg.db_path
            if os.path.isabs(self.cfg.db_path)
            else os.path.join(BASE_DIR, self.cfg.db_path)
        )
        # Initialize SQLite database connection
        # Actually could be better to use SQLDatabaseToolkit here
        if not os.path.exists(db_full_path):
            extract_entities_in_sql_db(
                self.embedding_model,
                self.cfg,
                json_data,
                db_full_path,
                faiss_index_path,
            )
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_full_path}")
        self.sql_chain = SQLDatabaseChain.from_llm(self.llm, self.db, verbose=True)

        # Initialize tools
        # TODO: Use here the decsription parameter as putting the info in the prompt is old practices.
        tools = [
            StructuredTool.from_function(self.semantic_search_tool),
            StructuredTool.from_function(self.query_sql_tool),
        ]

        # Create the agent
        self.graph = create_react_agent(self.llm, tools=tools)
        self.ask(question)

    def semantic_search_tool(self, query: str) -> str:
        """Retrieve relevant documents based on semantic similarity."""
        results = self.vector_store.similarity_search(query, k=2)
        return "\n".join([result.page_content for result in results])

    def query_sql_tool(self, question: str) -> str:
        """Answer questions by querying the SQL database."""
        return self.sql_chain.invoke({"query": question})

    def print_stream(self, stream):
        """Print the stream of messages."""
        for s in stream:
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

    def ask(self, user_input=None):
        """Ask a question and get a response."""
        # Refined prompt for decision-making
        refined_prompt = f"""
        Task: Answer the user's question using the most appropriate method: either by querying the SQL database or by performing a semantic search on the document content.

        ### Available Methods:
        1. **SQL Query**: Use this when the information is structured and can be retrieved using the database schema provided below. Generate a valid SQL query and execute it against the database.
        2. **Semantic Search**: Use this when the information is better retrieved by finding relevant document excerpts based on semantic similarity.

        ### SQL Database Schema:
        Table: entity_data
        Columns:
          - company (TEXT): Name of the company.
          - year (INTEGER): Year of the report.
          - revenue (REAL): Total revenue in the given year.
          - risks (TEXT): Summary of risks.
          - human_capital (INTEGER): Total number of employees.

        ### Rules for SQL Queries:
        1. Only use the column names and data types provided in the schema.
        2. Ensure valid SQL syntax for SQLite.
        3. Use appropriate filters (e.g., `year`, `company`) when applicable.
        4. Use SQL functions like `SUM`, `AVG`, or `GROUP BY` for aggregations if necessary.

        ### Semantic Search Rules:
        1. Perform semantic search when the user's question requires insights or contextual information not available in the SQL database.
        2. Retrieve and present relevant document excerpts ranked by semantic similarity.
        3. Ensure the retrieved content directly addresses the user's question.

        ### Decision-Making Guidelines:
        1. If the user's question explicitly involves structured data (e.g., "top companies by revenue" or "total employees in 2021"), prioritize SQL.
        2. If the user's question requires understanding of context or reasoning (e.g., "What are Amazon's main risks?" or "Summarize Apple's report for 2021"), use semantic search.
        3. When in doubt, start with semantic search and complement the answer with SQL data if applicable.

        User's Question:
        {user_input or self.question}
        """

        # Feed the refined prompt to the agent
        self.print_stream(
            self.graph.stream(
                {"messages": [("user", refined_prompt)]}, stream_mode="values"
            )
        )
