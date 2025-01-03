from document_assistant_agent.data_pipelines.pdf_text_extract import PDFTextExtractor
import os


relative_path = "document_assistant_agent/raw_documents/prepared/Amazon/"
full_path = os.path.abspath(relative_path)
extractor = PDFTextExtractor(
    full_path,
    device="cpu",
)
all_results = extractor.process_all_pdfs()
print(all_results)
