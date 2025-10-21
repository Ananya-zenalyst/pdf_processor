import pdfplumber
from itertools import groupby
import logging

# Configure logger
logger = logging.getLogger(__name__)

def extract_text_with_position(pdf_path):
    """
    Extracts text from a PDF, intelligently grouping words into lines and lines
    into coherent paragraphs based on their spatial proximity and alignment.
    This version is optimized for multi-column layouts.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        dict: A dictionary where keys are page numbers (e.g., "page_1") and
              values are lists of extracted text block dictionaries, each
              containing content and a bounding box.
    """
    logger.info(f"Starting text extraction from: {pdf_path}")
    all_text_data = {}

    with pdfplumber.open(pdf_path) as pdf:
        logger.info(f"PDF opened with pdfplumber. Total pages: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages):
            page_number = i + 1
            logger.debug(f"Processing page {page_number} for text extraction")
            
            # Extract words with their precise bounding boxes
            words = page.extract_words(x_tolerance=1, y_tolerance=1, keep_blank_chars=False)
            logger.debug(f"Page {page_number}: Found {len(words)} words")

            if not words:
                logger.debug(f"Page {page_number}: No words found")
                all_text_data[f"page_{page_number}"] = []
                continue

            # Group words into lines based on their vertical position ('top')
            lines = []
            # Group words by their 'top' coordinate to form lines
            for _, g in groupby(words, key=lambda w: w['top']):
                line_words = sorted(list(g), key=lambda w: w['x0'])
                if line_words:
                    # Create a bounding box for the entire line
                    line_bbox = (
                        min(w['x0'] for w in line_words),
                        min(w['top'] for w in line_words),
                        max(w['x1'] for w in line_words),
                        max(w['bottom'] for w in line_words),
                    )
                    lines.append({
                        "text": " ".join(w['text'] for w in line_words),
                        "bbox": line_bbox
                    })

            if not lines:
                logger.debug(f"Page {page_number}: No lines formed")
                all_text_data[f"page_{page_number}"] = []
                continue

            logger.debug(f"Page {page_number}: Formed {len(lines)} lines")

            # Group lines into paragraphs
            paragraphs = []
            if lines:
                current_para_lines = [lines[0]]
                for next_line in lines[1:]:
                    prev_line = current_para_lines[-1]
                    prev_bbox = prev_line['bbox']
                    next_bbox = next_line['bbox']
                    
                    # Vertical proximity check
                    vertical_gap = next_bbox[1] - prev_bbox[3]
                    line_height = prev_bbox[3] - prev_bbox[1]
                    is_vertically_close = vertical_gap < (line_height * 0.75)

                    # Horizontal overlap check (crucial for multi-column layouts)
                    horizontal_overlap = max(0, min(prev_bbox[2], next_bbox[2]) - max(prev_bbox[0], next_bbox[0]))
                    is_horizontally_aligned = horizontal_overlap > 0

                    if is_vertically_close and is_horizontally_aligned:
                        current_para_lines.append(next_line)
                    else:
                        # Finalize the current paragraph
                        para_bbox = (
                            min(l['bbox'][0] for l in current_para_lines),
                            min(l['bbox'][1] for l in current_para_lines),
                            max(l['bbox'][2] for l in current_para_lines),
                            max(l['bbox'][3] for l in current_para_lines),
                        )
                        paragraphs.append({
                            "type": "text",
                            "content": " ".join(l['text'] for l in current_para_lines),
                            "bbox": para_bbox
                        })
                        current_para_lines = [next_line]
                
                # Add the last paragraph
                para_bbox = (
                    min(l['bbox'][0] for l in current_para_lines),
                    min(l['bbox'][1] for l in current_para_lines),
                    max(l['bbox'][2] for l in current_para_lines),
                    max(l['bbox'][3] for l in current_para_lines),
                )
                paragraphs.append({
                    "type": "text",
                    "content": " ".join(l['text'] for l in current_para_lines),
                    "bbox": para_bbox
                })

            all_text_data[f"page_{page_number}"] = paragraphs
            logger.debug(f"Page {page_number}: Created {len(paragraphs)} paragraphs")

    logger.info(f"Text extraction completed. Processed {len(all_text_data)} pages")
    return all_text_data

