import fitz  # PyMuPDF
import pdfplumber
import pytesseract
import os
import io
from PIL import Image

# Ensure the output directory for extracted images exists
OUTPUT_IMAGE_DIR = "uploads/extracted_images"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

def extract_images_with_ocr_and_position(pdf_path):
    """
    Extracts images from each page of a PDF, performs OCR, and also extracts
    any digital text found within the image's bounding box (useful for charts).
    Returns the data along with each image's precise bounding box.
    """
    all_image_data = {}
    
    # Use PyMuPDF for robust image object detection
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF with PyMuPDF: {e}")
        return {}

    # Use pdfplumber in parallel to access text within specific areas
    with pdfplumber.open(pdf_path) as plumber_pdf:
        if len(doc) != len(plumber_pdf.pages):
             print("Warning: Page count mismatch between PyMuPDF and pdfplumber. Digital text extraction in charts may be affected.")

        for i in range(len(doc)):
            page_number = i + 1
            fitz_page = doc.load_page(i)
            plumber_page = plumber_pdf.pages[i] if i < len(plumber_pdf.pages) else None
            images_on_page = []

            # get_image_info is the most reliable way to find all image instances
            image_info_list = fitz_page.get_image_info(xrefs=True)

            for img_index, img_info in enumerate(image_info_list):
                bbox = img_info['bbox']
                xref = img_info['xref']

                if xref == 0: continue # Skip inline image masks

                digital_text_in_area = ""
                if plumber_page:
                    try:
                        # Crop the page to the image's bounding box to find any text inside it
                        cropped_page = plumber_page.crop(bbox)
                        digital_text_in_area = cropped_page.extract_text(x_tolerance=2, y_tolerance=2)
                    except Exception as e:
                        print(f"Could not extract digital text for image on page {page_number}: {e}")

                ocr_text = ""
                image_path = None
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    image_filename = f"page{page_number}_img{img_index + 1}.{image_ext}"
                    image_path = os.path.join(OUTPUT_IMAGE_DIR, image_filename)

                    img = Image.open(io.BytesIO(image_bytes))
                    img.save(image_path)
                    
                    # Perform OCR on the saved image file
                    ocr_text = pytesseract.image_to_string(img)
                except Exception as e:
                    print(f"Error processing image with xref {xref} on page {page_number}: {e}")
                
                images_on_page.append({
                    "type": "image",
                    "image_path": image_path,
                    "ocr_text": ocr_text.strip() if ocr_text else "",
                    "digital_text": digital_text_in_area.strip() if digital_text_in_area else "",
                    "bbox": bbox
                })
            
            all_image_data[f"page_{page_number}"] = images_on_page
    
    doc.close()
    return all_image_data

