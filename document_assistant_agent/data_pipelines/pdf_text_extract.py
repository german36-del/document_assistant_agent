from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import fitz  # PyMuPDF for PDF processing
import os
import re
import json


class PDFTextExtractor:
    def __init__(self, pdf_files_path, device):
        self.pdf_files_path = pdf_files_path
        self.processor = DonutProcessor.from_pretrained(
            "naver-clova-ix/donut-base-finetuned-cord-v2"
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "naver-clova-ix/donut-base-finetuned-cord-v2"
        )
        self.device = device
        self.model.to(device)

    def extract_text_from_pdf(self, pdf_path):
        # Open the PDF and extract each page as an image
        doc = fitz.open(pdf_path)
        results = []
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        for page_number in range(len(doc)):
            # Render page as image
            page = doc.load_page(page_number)
            pix = page.get_pixmap()
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Preprocess the image for the Donut model
            pixel_values = self.processor(image, return_tensors="pt").pixel_values

            # Generate and decode the output
            outputs = self.model.generate(
                pixel_values.to(self.device),
                decoder_input_ids=decoder_input_ids.to(self.device),
                max_length=self.model.decoder.config.max_position_embeddings,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(
                self.processor.tokenizer.pad_token, ""
            )
            sequence = re.sub(
                r"<.*?>", "", sequence, count=1
            ).strip()  # remove first task start token

            parsed_dict = self.processor.token2json(sequence)
            # print the prettified dictionary
            prettified_dict = json.dumps(parsed_dict, indent=4)
            print(prettified_dict)
            results.append({"page": page_number + 1, "content": prettified_dict})

        doc.close()
        return results

    def process_all_pdfs(self):
        results = {}
        for pdf_file in os.listdir(self.pdf_files_path):
            if pdf_file.endswith(".pdf"):
                full_path = os.path.join(self.pdf_files_path, pdf_file)
                results[pdf_file] = self.extract_text_from_pdf(full_path)
        return results


# Example usage:
# extractor = PDFTextExtractor("document_assistant_agent/raw_documents/prepared/Amazon")
# all_results = extractor.process_all_pdfs()
# print(all_results)
