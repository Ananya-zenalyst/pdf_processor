import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
import json

def extract_images_and_ocr(pdf_path, output_dir_name='extracted_images'):
    """
    Extracts images from a PDF, performs OCR, and saves only useful images.

    Args:
        pdf_path (str): The path to the PDF file.
        output_dir_name (str): The name of the directory to save extracted images.

    Returns:
        str: A JSON string containing OCR text from images per page.
    """
    pdf_dir = os.path.dirname(pdf_path)
    full_output_dir = os.path.join(pdf_dir, output_dir_name)
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)

    extracted_data = {}
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page_images = []
            for img_index, img in enumerate(pdf_document.get_page_images(page_num)):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                
                pil_image = Image.open(io.BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(pil_image).strip()
                
                # Only process and save images that contain actual text
                if ocr_text:
                    image_filename = f"page{page_num+1}_img{img_index+1}.{base_image['ext']}"
                    image_path = os.path.join(full_output_dir, image_filename)
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    page_images.append({
                        "image_path": image_path,
                        "ocr_text": ocr_text
                    })
            
            if page_images:
                extracted_data[f'page_{page_num+1}'] = page_images
    except Exception as e:
        return json.dumps({"error": f"An error occurred during image extraction/OCR: {e}"}, indent=4)

    return json.dumps(extracted_data, indent=4)

