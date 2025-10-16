import sys
import os
import json
from extractors.text_extractor import extract_text_with_position
from extractors.table_extractor import extract_tables_with_position
from extractors.image_extractor import extract_images_with_ocr_and_position
import fitz  # PyMuPDF

def _is_bbox_inside(inner_bbox, outer_bbox, tolerance=1.0):
    """
    Checks if the inner_bbox is located inside the outer_bbox.
    A small tolerance is added to handle minor detection discrepancies.
    """
    i_x0, i_top, i_x1, i_bottom = inner_bbox
    o_x0, o_top, o_x1, o_bottom = outer_bbox
    
    return (i_x0 >= o_x0 - tolerance and
            i_top >= o_top - tolerance and
            i_x1 <= o_x1 + tolerance and
            i_bottom <= o_bottom + tolerance)

def get_document_layout(pdf_path):
    """
    Analyzes the PDF to extract all content, de-duplicates text found within
    tables and charts, sorts the final content by visual order, and formats
    the output with page and element numbers as requested.
    """
    final_output = {}

    print("Step 1: Extracting tables...")
    table_content_by_page = extract_tables_with_position(pdf_path)
    
    print("Step 2: Extracting images and charts...")
    image_content_by_page = extract_images_with_ocr_and_position(pdf_path)
    
    print("Step 3: Extracting and filtering text blocks...")
    text_content_by_page = extract_text_with_position(pdf_path)

    try:
        with fitz.open(pdf_path) as pdf_doc:
            num_pages = len(pdf_doc)
    except Exception as e:
        return {"error": f"Could not open PDF to get page count: {e}"}

    # Process each page to assemble, filter, sort, and number the content
    for i in range(num_pages):
        page_number = i + 1
        page_key = f"page_{page_number}"
        
        page_elements = []
        
        # Get all structured elements for the current page
        tables_on_page = table_content_by_page.get(page_key, [])
        images_on_page = image_content_by_page.get(page_key, [])
        text_on_page = text_content_by_page.get(page_key, [])
        
        page_elements.extend(tables_on_page)
        page_elements.extend(images_on_page)
        
        # --- Critical De-duplication Step ---
        # Filter out any raw text blocks that are already contained within a table or chart
        for text_block in text_on_page:
            is_redundant = False
            # Check if text is inside any table's bounding box
            for table in tables_on_page:
                if _is_bbox_inside(text_block['bbox'], table['bbox']):
                    is_redundant = True
                    break
            if is_redundant:
                continue
            
            # Check if text is inside any image/chart's bounding box
            for image in images_on_page:
                if _is_bbox_inside(text_block['bbox'], image['bbox']):
                    is_redundant = True
                    break
            if is_redundant:
                continue
            
            # If the text is not redundant, add it to our list of elements
            page_elements.append(text_block)

        # Sort all unique elements by their position (top-to-bottom, then left-to-right)
        page_elements.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        # --- Final Numbering and Formatting ---
        page_output = []
        para_counter = 1
        table_counter = 1
        image_counter = 1

        for element in page_elements:
            element_type = element.get("type")
            
            # Create a clean copy of the element, removing the internal bbox
            clean_element = element.copy()
            del clean_element['bbox']

            if element_type == "text":
                clean_element['page_number'] = page_number
                clean_element['paragraph_number'] = para_counter
                para_counter += 1
            elif element_type == "table":
                clean_element['page_number'] = page_number
                clean_element['table_number'] = table_counter
                table_counter += 1
            elif element_type == "image":
                clean_element['page_number'] = page_number
                clean_element['image_number'] = image_counter
                image_counter += 1
            
            page_output.append(clean_element)

        final_output[page_key] = page_output

    return final_output

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)

    pdf_file_path = sys.argv[1]
    
    if not os.path.exists(pdf_file_path):
        print(f"Error: PDF file not found at {pdf_file_path}")
        sys.exit(1)

    print(f"Analyzing PDF: {pdf_file_path}")
    
    structured_content = get_document_layout(pdf_file_path)

    print("\n--- FINAL EXTRACTED JSON (Spatially Ordered, De-duplicated, and Numbered) ---")
    print(json.dumps(structured_content, indent=2, ensure_ascii=False))

