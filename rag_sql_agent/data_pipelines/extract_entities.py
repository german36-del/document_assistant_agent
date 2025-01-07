import os
import json
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from rag_sql_agent.utils.loaders import load_faiss_index
from rag_sql_agent.utils import LOGGER
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd
from collections import defaultdict
import sqlite3


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Define entity schema
class RevenueEntity(BaseModel):
    revenue: float = Field(
        description="Total income from goods sold or services provided."
    )
    revenue_reasoning: str = Field(
        description="Text from the document used to infer the revenue value."
    )
    revenue_unit: str = Field(description="Unit of revenue using ISO alphabetic code.")
    revenue_unit_reasoning: str = Field(
        description="Text used to infer the revenue unit."
    )


class RisksEntity(BaseModel):
    risks: str = Field(description="Summary of risks.")
    risks_reasoning: str = Field(description="Text used to infer the risks.")


class HumanCapitalEntity(BaseModel):
    human_capital: int = Field(description="Total number of employees.")
    human_capital_reasoning: str = Field(
        description="Text used to infer the human capital."
    )


entity_schema = {
    "revenue": RevenueEntity,
    "risks": RisksEntity,
    "human_capital": HumanCapitalEntity,
}

# Define entity configuration
entity_list = {
    "revenue": {
        "description": "Total income from goods sold or services.",
        "rag_query": "What is the total revenue for {company} in {year}?",
    },
    "risks": {
        "description": "Summary of risks.",
        "rag_query": "What are the main risks for {company} in {year}?",
    },
    "human_capital": {
        "description": "Total number of employees.",
        "rag_query": "What is the total number of employees for {company} in {year}?",
    },
}

# Define examples for few-shot learning
example_pairs = {
    "revenue": [
        {
            "document_excerpts": "Page 20 - After the 10% increase in the number of customers, the sales for Company Inc in 2019 was $513,983 million.",
            "json_output": {
                "revenue": 324483000000,
                "revenue_reasoning": "The sales for Company Inc in 2019 was $324,483 million.",
                "revenue_unit": "USD",
                "revenue_unit_reasoning": "The financial report is in US dollars as stated on page 20.",
            },
        }
    ],
    "human_capital": [
        {
            "document_excerpts": "Despite the COVID-19 pandemic, In 2019, Company Inc employed 349,329 employees worldwide.",
            "json_output": {
                "human_capital": 349329,
                "human_capital_reasoning": "In 2019, Company Inc employed 349,329 employees worldwide.",
            },
        }
    ],
    "risks": [
        {
            "document_excerpts": """Competition continues to intensify, including with the development of new business models and the entry of new and well-funded competitors.""",
            "json_output": {
                "risks": "The main risks are: \n* Competition from new entrants\n* Increased competition because of new technologies, ",
                "risks_reasoning": "Competition continues to intensify, including the development of new business models.",
            },
        }
    ],
}


# LLM prompt
ENTITY_EXTRACTION_PROMPT_TEMPLATE = """
Human: Extract the information described by the JSON schema inside the <schema></schema> XML tags from the documents inside <documents></documents> XML tags.
Follow the rules inside the <rules></rules> XML tags during extraction:
<rules>
1. You must output a valid JSON.
2. You must extract the value for each field from the text inside <documents></documents>, and the value must match the description and type in the JSON schema.
3. Expand numbers into full digits format: example 1: 212,765,000,000 becomes 212765000000, example 2: $469.822 million becomes 469822000, example 3: 132,452 people becomes 132452.
4. Don't use commas as thousands separators in the numbers you extract. For example, 212,765 must be written as 212765.
5. Consider the context inside <context></context> XML tags.
6. If the document does not contain the value, put null.
</rules>

The JSON schema inside the <schema></schema> XML tags contains the information to extract:
<schema>
{serialized_json_schema}
</schema>

Extract information from the documents inside <documents></documents> XML tags below:
<documents>
{document_excerpts}
</documents>

Use the metadata inside the <context></context> XML tags when relevant to assist you during extraction:
<context>
The company is {company}.
The year of the financial report is {year}.
</context>

Follow the extraction examples inside the <examples></examples> XML tags below:
<examples>
{few_shot_examples}
</examples>

Only write the JSON output inside <json></json> XML tags without further explanation.

\n\nAssistant: <json>\n
"""


# Generate few-shot examples
def generate_few_shot_examples(entity_name: str, examples: List[Dict[str, Any]]):
    example_template = """
    Example {index}: Given the information inside <schema> and <documents>, the correct output is inside <json> below:
    
    <schema>
    {serialized_json_schema}
    </schema>
    
    <documents>
    {document_excerpts}
    </documents>
    
    Correct output:
    <json>
    {json_output}
    </json>
    """
    serialized_json_schema = json.dumps(entity_schema[entity_name].schema(), indent=2)
    combined_examples = "\n".join(
        example_template.format(
            index=idx + 1,
            serialized_json_schema=serialized_json_schema,
            document_excerpts=example["document_excerpts"],
            json_output=json.dumps(example["json_output"], indent=2),
        )
        for idx, example in enumerate(examples)
    )
    return combined_examples


# Extract entity
def extract_entity(
    entity, metadata, chunks, entity_schema, few_shot_examples, llm_chain
):
    """Extract entity information using the specified rules."""
    document_excerpts = "\n".join([chunk.page_content for chunk in chunks])
    serialized_json_schema = json.dumps(entity_schema[entity].schema(), indent=2)
    prompt = ENTITY_EXTRACTION_PROMPT_TEMPLATE.format(
        serialized_json_schema=serialized_json_schema,
        document_excerpts=document_excerpts,
        company=metadata["company"],
        year=metadata["year"],
        few_shot_examples=few_shot_examples,
    )
    result = llm_chain.predict(
        serialized_json_schema=serialized_json_schema,
        document_excerpts=document_excerpts,
        company=metadata["company"],
        year=metadata["year"],
        few_shot_examples=few_shot_examples,
    )
    return result


def extract_entities_in_sql_db(
    embedding_model, cfg, json_data, full_db_path, faiss_index_path
):
    faiss = load_faiss_index(
        embedding_model,
        index_path=faiss_index_path,
        allow_dangerous_deserialization=True,
    )
    relative_path = os.path.join(
        cfg.documents_download_folder, "prepared", "metadata.json"
    )
    full_path = os.path.abspath(relative_path)
    full_metadata = load_json(full_path)
    llm = ChatOpenAI(model=cfg.openai_model, temperature=0)
    llm_chain = PromptTemplate.from_template(ENTITY_EXTRACTION_PROMPT_TEMPLATE) | llm

    few_shot_examples = {
        entity: generate_few_shot_examples(entity, example_pairs[entity])
        for entity in entity_list
    }

    extracted_entities = []
    for idx, document in enumerate(json_data):
        metadata = full_metadata[idx]  # Adjust as needed
        for entity in entity_list:
            LOGGER.info("Searching in the database possible solutions....")
            chunks = faiss.similarity_search(
                entity_list[entity]["rag_query"].format(
                    company=metadata["company"], year=metadata["year"]
                ),
                k=5,
            )
            result = extract_entity(
                entity,
                metadata,
                chunks,
                entity_schema,
                few_shot_examples[entity],
                llm_chain,
            )
            extracted_entities.append(
                {"entity_type": entity, "result": result, **metadata}
            )
    # Initialize a dictionary to hold the aggregated data
    aggregated_data = []

    # Process each record and aggregate the data
    for record in extracted_entities:
        # Parse the result field into a dictionary
        try:
            result = json.loads(record["result"])
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for record: {record}")
            continue

        # Find the matching entry in aggregated data
        entry = next(
            (
                item
                for item in aggregated_data
                if item["company"] == record["company"]
                and item["year"] == record["year"]
            ),
            None,
        )

        # If no entry exists, create a new one
        if not entry:
            entry = {
                "company": record["company"],
                "year": record["year"],
                "source_doc": record.get(
                    "doc_url", record.get("local_pdf_path", "Unknown")
                ),
                "revenue": None,
                "revenue_reasoning": None,
                "revenue_unit": None,
                "revenue_unit_reasoning": None,
                "risks": None,
                "risks_reasoning": None,
                "human_capital": None,
                "human_capital_reasoning": None,
            }
            aggregated_data.append(entry)

        # Update the entry with entity-specific data
        if record["entity_type"] == "revenue":
            entry.update(
                {
                    "revenue": result.get("revenue"),
                    "revenue_reasoning": result.get("revenue_reasoning"),
                    "revenue_unit": result.get("revenue_unit"),
                    "revenue_unit_reasoning": result.get("revenue_unit_reasoning"),
                }
            )
        elif record["entity_type"] == "risks":
            entry.update(
                {
                    "risks": result.get("risks"),
                    "risks_reasoning": result.get("risks_reasoning"),
                }
            )
        elif record["entity_type"] == "human_capital":
            entry.update(
                {
                    "human_capital": result.get("human_capital"),
                    "human_capital_reasoning": result.get("human_capital_reasoning"),
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(aggregated_data)

    # Optionally save to CSV
    df.to_csv("aggregated_entities_table.csv", index=False)

    def save_to_sqlite(dataframe, db_name="local_data.db", table_name="entity_data"):
        # Establish a connection to the SQLite database
        conn = sqlite3.connect(db_name)

        # Save the DataFrame to the specified table
        dataframe.to_sql(table_name, conn, if_exists="replace", index=False)

        # Close the connection
        conn.close()

    # Call the function
    save_to_sqlite(df, db_name=full_db_path)
