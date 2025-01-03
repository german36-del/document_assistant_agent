import json
import os
from pypdf import PdfReader, PdfWriter
import shutil


def keep_relevant_pages_in_pdf(input_pdf_path, output_pdf_path, pages):
    input_pdf = PdfReader(input_pdf_path)
    print(f"Number of pages is {len(input_pdf.pages)}")
    print(f"Relevant pages are {pages}")
    output_pdf = PdfWriter()

    for page_num in pages:
        output_pdf.add_page(input_pdf.pages[page_num - 1])

    with open(output_pdf_path, "wb") as f:
        output_pdf.write(f)


def save_json(json_data, file_path):
    with open(file_path, "w") as f:
        json.dump(json_data, f)


def keep_relevant_pages_in_pdfs(
    raw_base_directory, prepared_base_directory, docs_mapping
):
    metadata = []
    # Create the base directory if it doesn't exist
    if not os.path.exists(prepared_base_directory):
        os.makedirs(prepared_base_directory)

    for company, docs in docs_mapping.items():
        raw_company_directory = os.path.join(raw_base_directory, company)
        prepared_company_directory = os.path.join(prepared_base_directory, company)

        # Create a directory for the company if it doesn't exist
        if not os.path.exists(prepared_company_directory):
            os.makedirs(prepared_company_directory)

        for doc_info in docs:
            doc_url = doc_info["doc_url"]
            year = doc_info["year"]
            pages = doc_info.get("pages", [])
            if not doc_url:
                continue

            current_metadata = {}
            current_metadata["company"] = company
            current_metadata["year"] = year
            current_metadata["doc_url"] = doc_url

            # Construct the filename based on the year and the URL
            filename = f"annual_report_{year}.pdf"
            input_pdf_path = os.path.join(raw_company_directory, filename)
            output_pdf_path = os.path.join(prepared_company_directory, filename)

            current_metadata["local_pdf_path"] = output_pdf_path

            if not pages:
                # When page numbers are not defined, we assume the user wants
                # to process the full file, therefore, copy it as is
                # to the prepared folder
                shutil.copyfile(input_pdf_path, output_pdf_path)
                metadata.append(current_metadata)
                continue

            relevant_pages = doc_info["pages"]
            current_metadata["pages_kept"] = relevant_pages

            # Skip empty URLs

            keep_relevant_pages_in_pdf(input_pdf_path, output_pdf_path, relevant_pages)

            metadata.append(current_metadata)

    save_json(metadata, os.path.join(prepared_base_directory, "metadata.json"))

    return True


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
raw_base_directory = "raw_documents"

prepared_base_directory = os.path.join(raw_base_directory, "prepared/")
keep_relevant_pages_in_pdfs(raw_base_directory, prepared_base_directory, docs_mapping)
