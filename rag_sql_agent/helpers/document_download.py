import os
import requests


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
