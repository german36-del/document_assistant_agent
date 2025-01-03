import os
import requests

raw_base_directory = "raw_documents"

if not os.path.exists(raw_base_directory):
    os.makedirs(raw_base_directory)

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


def download_pdf_files(base_directory, docs_mapping, headers):
    # Create the base directory if it doesn't exist
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    for company, docs in docs_mapping.items():
        company_directory = os.path.join(base_directory, company)

        # Create a directory for the company if it doesn't exist
        if not os.path.exists(company_directory):
            os.makedirs(company_directory)

        for doc_info in docs:
            doc_url = doc_info["doc_url"]
            year = doc_info["year"]

            # Skip empty URLs
            if not doc_url:
                continue

            # Construct the filename based on the year and the URL
            filename = f"annual_report_{year}.pdf"
            file_path = os.path.join(company_directory, filename)

            # Check if the file already exists
            if os.path.exists(file_path):
                print(f"{filename} already exists for {company}")
            else:
                # Download the document
                response = requests.get(doc_url, headers=headers)

                if response.status_code == 200:
                    with open(file_path, "wb") as file:
                        file.write(response.content)
                    print(f"Downloaded {filename} for {company}")
                else:
                    print(
                        f"Failed to download {filename} for {company}"
                        f" (Status Code: {response.status_code})"
                    )


# Define user-agent and headers to mimic a browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
download_pdf_files(raw_base_directory, docs_mapping, headers)
