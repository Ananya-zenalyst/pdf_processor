import sys
import json
import pdfplumber
import os

# Allow Python to find our extractor modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extractors.text_extractor import extract_text
from extractors.table_extractor import extract_tables
from extractors.image_extractor import extract_images_and_ocr

def analyze_and_extract(pdf_path):
    """
    Analyzes a PDF to decide which content types are present and
    routes them to the appropriate extractors.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: A JSON string with all extracted content.
    """
    final_extraction = {}
    has_text, has_tables, has_images = False, False, False

    # 1. Analyze the PDF to determine which extractors to run
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                if page.extract_text(x_tolerance=2).strip(): has_text = True
                if page.find_tables(): has_tables = True
                if page.images: has_images = True
                if has_text and has_tables and has_images: break
    except Exception as e:
        return json.dumps({"error": f"Failed to analyze PDF: {e}"}, indent=4)

    print("PDF Analysis Complete:")
    print(f"- Contains Text: {has_text}, Contains Tables: {has_tables}, Contains Images: {has_images}")
    print("-" * 20)

    # 2. Route to extractors based on the analysis
    if has_text:
        text_data = json.loads(extract_text(pdf_path))
        if text_data: final_extraction["text_content"] = text_data

    if has_tables:
        table_data = json.loads(extract_tables(pdf_path))
        if table_data: final_extraction["table_content"] = table_data

    if has_images:
        image_data = json.loads(extract_images_and_ocr(pdf_path))
        if image_data: final_extraction["image_content"] = image_data

    return json.dumps(final_extraction, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf_file>")
        sys.exit(1)
    
    pdf_file_path = sys.argv[1]
    if not os.path.exists(pdf_file_path):
        print(f"Error: File not found at '{pdf_file_path}'")
        sys.exit(1)

    extracted_json = analyze_and_extract(pdf_file_path)
    print("\n--- FINAL EXTRACTED JSON ---")
    print(extracted_json)

