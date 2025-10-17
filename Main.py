import sys
import os
import json
import cv2
import numpy as np
from extractors.text_extractor import extract_text_with_position
from extractors.table_extractor import extract_tables_with_position
from extractors.image_extractor import extract_images_with_ocr_and_position
import fitz  # PyMuPDF
from typing import Dict, List, Tuple, Any

def analyze_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """
    Uses OpenCV to analyze PDF content and identify regions containing:
    - Tables (structured data with lines/grids)
    - Images (photos, charts, diagrams)
    - Text blocks (paragraphs, headers)

    Returns a detailed analysis of content types per page.
    """
    content_analysis = {
        'total_pages': 0,
        'page_analysis': {},
        'summary': {
            'has_tables': False,
            'has_images': False,
            'has_text': False,
            'table_pages': [],
            'image_pages': [],
            'text_pages': []
        }
    }

    try:
        doc = fitz.open(pdf_path)
        content_analysis['total_pages'] = len(doc)

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Convert page to image for OpenCV analysis
            mat = fitz.Matrix(2, 2)  # 2x zoom for better analysis
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Analyze the page
            page_info = analyze_page_with_opencv(image, page_num + 1)
            content_analysis['page_analysis'][f'page_{page_num + 1}'] = page_info

            # Update summary
            if page_info['has_tables']:
                content_analysis['summary']['has_tables'] = True
                content_analysis['summary']['table_pages'].append(page_num + 1)
            if page_info['has_images']:
                content_analysis['summary']['has_images'] = True
                content_analysis['summary']['image_pages'].append(page_num + 1)
            if page_info['has_text']:
                content_analysis['summary']['has_text'] = True
                content_analysis['summary']['text_pages'].append(page_num + 1)

        doc.close()

    except Exception as e:
        print(f"Error analyzing PDF: {e}")

    return content_analysis

def analyze_page_with_opencv(image: np.ndarray, page_num: int) -> Dict[str, Any]:
    """
    Analyzes a single page image using OpenCV to detect content types.
    """
    page_info = {
        'page_number': page_num,
        'has_tables': False,
        'has_images': False,
        'has_text': False,
        'table_regions': [],
        'image_regions': [],
        'text_regions': [],
        'confidence': {}
    }

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Detect tables using line detection
    table_info = detect_tables_opencv(gray, image)
    if table_info['tables_found']:
        page_info['has_tables'] = True
        page_info['table_regions'] = table_info['regions']
        page_info['confidence']['tables'] = table_info['confidence']

    # 2. Detect images/graphics
    image_info = detect_images_opencv(gray, image)
    if image_info['images_found']:
        page_info['has_images'] = True
        page_info['image_regions'] = image_info['regions']
        page_info['confidence']['images'] = image_info['confidence']

    # 3. Detect text regions
    text_info = detect_text_regions_opencv(gray, image)
    if text_info['text_found']:
        page_info['has_text'] = True
        page_info['text_regions'] = text_info['regions']
        page_info['confidence']['text'] = text_info['confidence']

    return page_info

def detect_tables_opencv(gray: np.ndarray, image: np.ndarray) -> Dict[str, Any]:
    """
    Detects table structures using OpenCV line and contour detection.
    """
    result = {
        'tables_found': False,
        'regions': [],
        'confidence': 0.0
    }

    # Apply adaptive threshold
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)

    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    # Extract lines
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    # Combine lines to get table structure
    table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # Find contours of potential tables
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area (tables should be reasonably sized)
    min_table_area = image.shape[0] * image.shape[1] * 0.002  # At least 0.2% of page
    max_table_area = image.shape[0] * image.shape[1] * 0.8    # At most 80% of page

    valid_tables = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_table_area < area < max_table_area:
            x, y, w, h = cv2.boundingRect(contour)

            # Additional checks for table-like structure
            aspect_ratio = w / h if h > 0 else 0
            if 0.5 < aspect_ratio < 10:  # Tables are usually not too narrow or too wide
                # Check for presence of intersecting lines (grid structure)
                roi_h = horizontal_lines[y:y+h, x:x+w]
                roi_v = vertical_lines[y:y+h, x:x+w]

                h_lines = cv2.countNonZero(roi_h)
                v_lines = cv2.countNonZero(roi_v)

                # Tables should have both horizontal and vertical lines
                if h_lines > 100 and v_lines > 100:
                    valid_tables.append({
                        'bbox': (x, y, x+w, y+h),
                        'area': area,
                        'line_density': (h_lines + v_lines) / (w * h),
                        'confidence': min(1.0, (h_lines + v_lines) / 1000)
                    })

    if valid_tables:
        result['tables_found'] = True
        result['regions'] = valid_tables
        result['confidence'] = sum(t['confidence'] for t in valid_tables) / len(valid_tables)

    return result

def detect_images_opencv(gray: np.ndarray, image: np.ndarray) -> Dict[str, Any]:
    """
    Detects image regions (photos, charts, diagrams) using OpenCV.
    """
    result = {
        'images_found': False,
        'regions': [],
        'confidence': 0.0
    }

    # Edge detection to find image boundaries
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for image-like regions
    min_image_area = image.shape[0] * image.shape[1] * 0.005  # At least 0.5% of page

    valid_images = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_image_area:
            x, y, w, h = cv2.boundingRect(contour)

            # Check if region has image-like properties
            roi = image[y:y+h, x:x+w]

            # Calculate color variance (images have more color variation)
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                color_std = np.std(roi)

                # Images typically have higher color variance than text
                if color_std > 30:
                    # Check aspect ratio (images are often square-ish or standard photo ratios)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.3 < aspect_ratio < 3:
                        valid_images.append({
                            'bbox': (x, y, x+w, y+h),
                            'area': area,
                            'color_variance': color_std,
                            'confidence': min(1.0, color_std / 100)
                        })

    if valid_images:
        result['images_found'] = True
        result['regions'] = valid_images
        result['confidence'] = sum(img['confidence'] for img in valid_images) / len(valid_images)

    return result

def detect_text_regions_opencv(gray: np.ndarray, image: np.ndarray) -> Dict[str, Any]:
    """
    Detects text regions using OpenCV.
    """
    result = {
        'text_found': False,
        'regions': [],
        'confidence': 0.0
    }

    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use morphological operations to connect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for text-like regions
    valid_text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Text regions have specific characteristics
        if area > 100:  # Minimum area for text
            aspect_ratio = w / h if h > 0 else 0

            # Text blocks are usually wider than tall
            if aspect_ratio > 0.5:
                # Check fill ratio (text regions are not completely filled)
                roi = binary[y:y+h, x:x+w]
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    fill_ratio = cv2.countNonZero(roi) / (w * h)

                    # Text regions have moderate fill ratio (not too sparse, not too dense)
                    if 0.1 < fill_ratio < 0.7:
                        valid_text_regions.append({
                            'bbox': (x, y, x+w, y+h),
                            'area': area,
                            'fill_ratio': fill_ratio,
                            'confidence': 0.8
                        })

    if valid_text_regions:
        result['text_found'] = True
        result['regions'] = valid_text_regions
        result['confidence'] = 0.8

    return result

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

def get_document_layout(pdf_path, use_opencv_analysis=True):
    """
    Analyzes the PDF to extract all content, de-duplicates text found within
    tables and charts, sorts the final content by visual order, and formats
    the output with page and element numbers as requested.

    Args:
        pdf_path: Path to the PDF file
        use_opencv_analysis: If True, uses OpenCV to pre-analyze content types for better extraction
    """
    final_output = {}

    # Step 0: Analyze content types using OpenCV (optional but recommended)
    if use_opencv_analysis:
        print("Step 0: Analyzing PDF content with OpenCV...")
        content_analysis = analyze_pdf_content(pdf_path)

        print(f"Content Analysis Summary:")
        print(f"  - Total pages: {content_analysis['total_pages']}")
        print(f"  - Pages with tables: {content_analysis['summary']['table_pages']}")
        print(f"  - Pages with images: {content_analysis['summary']['image_pages']}")
        print(f"  - Pages with text: {content_analysis['summary']['text_pages']}")

        # Only extract tables if tables are detected
        if content_analysis['summary']['has_tables']:
            print("\nStep 1: Extracting tables (tables detected)...")
            table_content_by_page = extract_tables_with_position(pdf_path)
        else:
            print("\nStep 1: Skipping table extraction (no tables detected)")
            table_content_by_page = {}

        # Only extract images if images are detected
        if content_analysis['summary']['has_images']:
            print("Step 2: Extracting images and charts (images detected)...")
            image_content_by_page = extract_images_with_ocr_and_position(pdf_path)
        else:
            print("Step 2: Skipping image extraction (no images detected)")
            image_content_by_page = {}

        # Always extract text as fallback
        print("Step 3: Extracting and filtering text blocks...")
        text_content_by_page = extract_text_with_position(pdf_path)
    else:
        # Traditional extraction without pre-analysis
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

def intelligent_pdf_extraction(pdf_path: str, output_format: str = 'json') -> Dict[str, Any]:
    """
    Intelligently extracts content from PDF using OpenCV analysis to route to appropriate extractors.
    This is the main controller that determines what type of content exists and calls the right extractor.

    Args:
        pdf_path: Path to the PDF file
        output_format: Output format ('json', 'excel', or 'both')

    Returns:
        Dictionary with extracted content organized by type
    """
    print("=" * 60)
    print("Starting Intelligent PDF Content Extraction")
    print("=" * 60)

    # Step 1: Analyze content with OpenCV
    print("\n📊 Analyzing PDF structure with OpenCV...")
    content_analysis = analyze_pdf_content(pdf_path)

    # Initialize result structure
    result = {
        'metadata': {
            'pdf_path': pdf_path,
            'total_pages': content_analysis['total_pages'],
            'content_types': {
                'has_tables': content_analysis['summary']['has_tables'],
                'has_images': content_analysis['summary']['has_images'],
                'has_text': content_analysis['summary']['has_text']
            }
        },
        'tables': None,
        'images': None,
        'text': None,
        'combined_output': None
    }

    # Step 2: Extract based on detected content types
    print(f"\n📋 Content Detection Results:")
    print(f"  ✓ Tables found: {content_analysis['summary']['has_tables']} (pages: {content_analysis['summary']['table_pages']})")
    print(f"  ✓ Images found: {content_analysis['summary']['has_images']} (pages: {content_analysis['summary']['image_pages']})")
    print(f"  ✓ Text found: {content_analysis['summary']['has_text']} (pages: {content_analysis['summary']['text_pages']})")

    # Extract tables if detected
    if content_analysis['summary']['has_tables']:
        print("\n🔍 Extracting tables with specialized table extractor...")
        try:
            from extractors.table_extractor import extract_tables_with_cv, extract_tables_to_excel

            if output_format in ['excel', 'both']:
                # Direct to Excel extraction for tables
                excel_output = extract_tables_to_excel(pdf_path, f"{pdf_path.replace('.pdf', '')}_tables.xlsx")
                result['tables'] = {
                    'format': 'excel',
                    'file_path': excel_output,
                    'pages': content_analysis['summary']['table_pages']
                }
                print(f"    ✓ Tables extracted to Excel: {excel_output}")
            else:
                # JSON extraction for tables
                tables_json = extract_tables_with_cv(pdf_path)
                result['tables'] = {
                    'format': 'json',
                    'data': json.loads(tables_json),
                    'pages': content_analysis['summary']['table_pages']
                }
                print(f"    ✓ Tables extracted to JSON format")
        except Exception as e:
            print(f"    ❌ Table extraction failed: {e}")
            result['tables'] = {'error': str(e)}

    # Extract images if detected
    if content_analysis['summary']['has_images']:
        print("\n🖼️ Extracting images with OCR...")
        try:
            images = extract_images_with_ocr_and_position(pdf_path)
            result['images'] = {
                'data': images,
                'pages': content_analysis['summary']['image_pages'],
                'count': sum(len(imgs) for imgs in images.values())
            }
            print(f"    ✓ {result['images']['count']} images extracted")
        except Exception as e:
            print(f"    ❌ Image extraction failed: {e}")
            result['images'] = {'error': str(e)}

    # Extract text content
    if content_analysis['summary']['has_text']:
        print("\n📝 Extracting text content...")
        try:
            text_content = extract_text_with_position(pdf_path)
            result['text'] = {
                'data': text_content,
                'pages': content_analysis['summary']['text_pages'],
                'paragraphs': sum(len(texts) for texts in text_content.values())
            }
            print(f"    ✓ {result['text']['paragraphs']} text blocks extracted")
        except Exception as e:
            print(f"    ❌ Text extraction failed: {e}")
            result['text'] = {'error': str(e)}

    # Step 3: Combine all content preserving layout
    if output_format in ['json', 'both']:
        print("\n🔄 Combining all content with layout preservation...")
        result['combined_output'] = get_document_layout(pdf_path, use_opencv_analysis=False)

    print("\n" + "=" * 60)
    print("✅ Intelligent extraction completed!")
    print("=" * 60)

    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf> [mode] [format]")
        print("  mode: 'intelligent' (default) or 'traditional'")
        print("  format: 'json' (default), 'excel', or 'both'")
        print("\nExamples:")
        print("  python main.py document.pdf                    # Intelligent extraction to JSON")
        print("  python main.py document.pdf intelligent excel  # Intelligent extraction with Excel output for tables")
        print("  python main.py document.pdf traditional        # Traditional extraction without OpenCV")
        sys.exit(1)

    pdf_file_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'intelligent'
    output_format = sys.argv[3] if len(sys.argv) > 3 else 'json'

    if not os.path.exists(pdf_file_path):
        print(f"Error: PDF file not found at {pdf_file_path}")
        sys.exit(1)

    print(f"Analyzing PDF: {pdf_file_path}")
    print(f"Mode: {mode}")
    print(f"Output format: {output_format}")

    if mode == 'intelligent':
        # Use intelligent extraction with OpenCV analysis
        result = intelligent_pdf_extraction(pdf_file_path, output_format)

        # Save results
        if output_format in ['json', 'both']:
            output_filename = pdf_file_path.replace('.pdf', '_intelligent_extraction.json')
            with open(output_filename, 'w', encoding='utf-8') as f:
                # Remove None values for cleaner JSON
                clean_result = {k: v for k, v in result.items() if v is not None}
                json.dump(clean_result, f, ensure_ascii=False, indent=2)
            print(f"\n📄 Results saved to {output_filename}")

        if output_format in ['excel', 'both'] and result.get('tables'):
            if result['tables'].get('format') == 'excel':
                print(f"\n📊 Tables saved to Excel: {result['tables'].get('file_path', 'N/A')}")
    else:
        # Traditional extraction
        structured_content = get_document_layout(pdf_file_path)

        # Save to JSON
        output_filename = pdf_file_path.replace('.pdf', '_layout.json')
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(structured_content, f, ensure_ascii=False, indent=2)

        print("\n--- FINAL EXTRACTED JSON (Spatially Ordered, De-duplicated, and Numbered) ---")
        print(f"Results saved to {output_filename}")

