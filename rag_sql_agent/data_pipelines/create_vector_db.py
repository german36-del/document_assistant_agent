from langchain_community.vectorstores import FAISS
from rag_sql_agent.data_pipelines.pdf_text_extract import PDFTextExtractor
from rag_sql_agent.helpers.document_download import download_pdf_files
from rag_sql_agent.helpers.keep_relevant_pages import keep_relevant_pages_in_pdfs
import json
import os

docs_mapping = {
    "Amazon": [
        {
            "doc_url": "https://s2.q4cdn.com/299287126/files/doc_financials/2023/ar/Amazon-2022-Annual-Report.pdf",
            "year": "2022",
            "pages": [15, 17, 18, 47, 48],
        },
        {
            "doc_url": "https://s2.q4cdn.com/299287126/files/doc_financials/2022/ar/Amazon-2021-Annual-Report.pdf",
            "year": "2021",
            "pages": [14, 16, 17, 18, 46, 47],
        },
        {"doc_url": "", "year": ""},
    ]
}


# Step 2: Flatten Data into Text Chunks
def prepare_text_chunks(data):
    chunks = []
    for file_name, pages in data.items():
        for page in pages:
            chunks.append(
                {
                    "content": page["content"],
                    "metadata": {"source": file_name, "page": page["page"]},
                }
            )
    return chunks


# Step 3: Generate Embeddings and Create FAISS Index
def create_faiss_index(chunks, embeddings_model, faiss_index_path):
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    vectorstore = FAISS.from_texts(texts, embeddings_model, metadatas=metadatas)
    vectorstore.save_local(faiss_index_path)
    return vectorstore


def create_vector_db(embeddings_model, cfg, faiss_index_path):
    if not os.path.exists(cfg.documents_download_folder):
        os.makedirs(cfg.documents_download_folder)
    # Define user-agent and headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    download_pdf_files(cfg.documents_download_folder, docs_mapping, headers)
    keep_relevant_pages_in_pdfs(
        cfg.documents_download_folder,
        os.path.join(cfg.documents_download_folder, "prepared/"),
        docs_mapping,
    )
    relative_path = os.path.join(cfg.documents_download_folder, "prepared", "Amazon")
    full_path = os.path.abspath(relative_path)
    extractor = PDFTextExtractor(full_path)
    all_results = extractor.process_all_pdfs()

    # Prepare text chunks
    chunks = prepare_text_chunks(all_results)

    # Create FAISS index
    vectorstore = create_faiss_index(chunks, embeddings_model, faiss_index_path)
    return all_results
