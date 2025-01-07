from langchain_community.document_loaders import PyPDFLoader
import os
import json


class PDFTextExtractor:
    def __init__(self, pdf_files_path):
        self.pdf_files_path = pdf_files_path

    def extract_text_from_pdf(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        results = []
        for doc in docs:
            results.append(
                {
                    "page": doc.metadata["page"],
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )
        return results

    def process_all_pdfs(self):
        results = {}
        for pdf_file in os.listdir(self.pdf_files_path):
            if pdf_file.endswith(".pdf"):
                full_path = os.path.join(self.pdf_files_path, pdf_file)
                results[pdf_file] = self.extract_text_from_pdf(full_path)
        return results

    def save_results_to_json(self, results, output_file="results.json"):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
