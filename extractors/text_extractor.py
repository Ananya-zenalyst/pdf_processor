import pdfplumber
import json

def extract_text(pdf_path):
    """
    Extracts text from each page of a PDF file using pdfplumber.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: A JSON string containing the extracted text per page.
    """
    extracted_data = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                # Only add the page if it has meaningful text content to avoid bloat
                if text and text.strip():
                    extracted_data[f'page_{i+1}'] = text.strip()
    except Exception as e:
        return json.dumps({"error": f"An error occurred during text extraction: {e}"}, indent=4)
    
    return json.dumps(extracted_data, indent=4)

