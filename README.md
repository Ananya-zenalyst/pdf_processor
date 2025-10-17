# Advanced PDF Data Extraction API

This project provides a powerful, multi-stage API for extracting text, tables, and images from PDF documents. It is designed to preserve the spatial layout of the original document and uses a combination of pdfplumber, camelot, and OpenCV for state-of-the-art extraction.

## Features

- **Spatially-Aware Extraction**: Reconstructs the top-to-bottom, left-to-right reading order of the PDF
- **Intelligent Text Grouping**: Correctly groups text into paragraphs, even in multi-column layouts
- **Advanced Table Extraction**: Multi-stage pipeline for maximum accuracy:
  - Camelot: For tables with clear borders
  - pdfplumber: For borderless tables based on text alignment
  - OpenCV: Computer vision for difficult or scanned tables
- **Chart Data Extraction**: Pulls both digital text and OCR'd text from charts and images
- **Content De-duplication**: Intelligently removes redundant text already part of extracted tables or charts
- **Asynchronous Processing**: Handles large files efficiently with background tasks
- **OpenCV Content Analysis**: Pre-analyzes PDFs to identify content types before extraction

## Prerequisites

Before you begin, you must have the following external software installed on your system:

- **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
- **Tesseract OCR**: Required for reading text from images
  - [Installation Guide for Tesseract](https://github.com/tesseract-ocr/tesseract)
- **Ghostscript**: Required dependency for camelot
  - [Download Ghostscript](https://www.ghostscript.com/download/gsdnld.html)
- **Java Runtime Environment**: Required dependency for tabula-py (used in advanced table extraction)
  - [Download Java](https://www.java.com/download/)

## Setup

### 1. Create Project Structure
Create the main folder `PDF_Extractor` with the following structure:
```
PDF_Extractor/
├── api.py                  # FastAPI application
├── main.py                 # Core extraction logic with OpenCV analysis
├── requirements.txt        # Python dependencies
├── extractors/            # Extraction modules
│   ├── __init__.py
│   ├── text_extractor.py   # Text extraction logic
│   ├── table_extractor.py  # Table extraction with multiple methods
│   └── image_extractor.py  # Image/chart extraction with OCR
└── uploads/               # Temporary file storage (created automatically)
```

### 2. Create a Virtual Environment (Recommended)
Navigate to the PDF_Extractor directory and run:
```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment
- **On macOS/Linux**: `source venv/bin/activate`
- **On Windows**: `venv\Scripts\activate`

### 4. Install Dependencies
With your virtual environment active:
```bash
pip install -r requirements.txt
```

## How to Run the API

### Start the Server
From the root `PDF_Extractor` directory:
```bash
uvicorn api:app --reload
```

The server will start on `http://127.0.0.1:8000`

### Access the API Documentation
Open your browser and navigate to `http://127.0.0.1:8000/docs` for the interactive FastAPI documentation.

## Application Flow

### Architecture Overview

The application follows a multi-layered architecture:

1. **API Layer** (`api.py`): FastAPI endpoints for handling HTTP requests
2. **Controller Layer** (`main.py`): Core logic for orchestrating extraction
3. **Extractor Modules** (`extractors/`): Specialized extraction components
4. **Background Processing**: Asynchronous task handling for large PDFs

### Processing Pipeline

```
User uploads PDF → API receives file → Background task starts
                                           ↓
                                   OpenCV Content Analysis
                                           ↓
                            Identifies content types per page:
                            - Tables (grid structures)
                            - Images (photos, charts)
                            - Text (paragraphs, headers)
                                           ↓
                          Parallel extraction based on content:
                      ↙                    ↓                    ↘
            Table Extractor        Image Extractor        Text Extractor
            (3-stage process)      (with OCR support)     (position-aware)
                      ↘                    ↓                    ↙
                              Content De-duplication
                              (removes redundant text)
                                           ↓
                              Spatial Layout Ordering
                              (top-to-bottom, left-to-right)
                                           ↓
                                 Final JSON/Excel Output
```

### Key Components

#### 1. OpenCV Content Analysis (`main.py`)
- Pre-analyzes PDF pages using computer vision
- Detects tables via line detection and grid patterns
- Identifies images through edge detection and color variance
- Locates text regions using morphological operations
- Optimizes extraction by only calling relevant extractors

#### 2. Table Extraction (`table_extractor.py`)
Three-stage approach for maximum accuracy:
- **Stage 1**: Camelot for bordered tables
- **Stage 2**: pdfplumber for borderless tables
- **Stage 3**: OpenCV fallback for complex/scanned tables
- Excel export with formatting preservation

#### 3. Image Extraction (`image_extractor.py`)
- Extracts embedded images from PDF
- Applies OCR to images for text extraction
- Handles charts and diagrams
- Preserves position information

#### 4. Text Extraction (`text_extractor.py`)
- Groups text into logical paragraphs
- Maintains spatial relationships
- Handles multi-column layouts
- Filters out redundant text from tables/images

### API Endpoints

#### `/extract/` (POST) - Asynchronous Processing
For large PDFs with background processing:
1. Upload PDF file
2. Receive task_id immediately
3. Poll `/status/{task_id}` for results
4. Get complete JSON when ready

#### `/extract-to-excel/` (POST) - Direct Excel Export
Extracts content and returns Excel file:
- Converts tables to Excel sheets
- Includes table-like text patterns
- Applies currency formatting
- Immediate download response

#### `/extract-tables-to-excel/` (POST) - Tables Only
Specialized CV-enhanced table extraction:
- Uses advanced OpenCV detection
- Finds tables that other methods might miss
- Direct Excel download

### Example API Usage

#### 1. Upload PDF for Asynchronous Processing
```bash
# Upload PDF
curl -X POST "http://127.0.0.1:8000/extract/" \
  -F "file=@document.pdf"

# Response
{
  "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "message": "File upload successful. Processing has started."
}
```

#### 2. Check Processing Status
```bash
# Check status
curl "http://127.0.0.1:8000/status/a1b2c3d4-e5f6-7890-1234-567890abcdef"

# Response (when complete)
{
  "status": "SUCCESS",
  "data": {
    "page_1": [
      {
        "type": "text",
        "content": "Report Title",
        "page_number": 1,
        "paragraph_number": 1
      },
      {
        "type": "table",
        "content": { ... table data ... },
        "page_number": 1,
        "table_number": 1
      }
    ]
  }
}
```

#### 3. Direct Excel Export
```bash
# Get Excel file directly
curl -X POST "http://127.0.0.1:8000/extract-to-excel/" \
  -F "file=@document.pdf" \
  -o extracted_content.xlsx
```

## Command-Line Usage

The `main.py` can also be run directly from command line:

### Intelligent Extraction (with OpenCV analysis)
```bash
# Default: JSON output with intelligent extraction
python main.py document.pdf

# Excel output for tables
python main.py document.pdf intelligent excel

# Both JSON and Excel output
python main.py document.pdf intelligent both
```

### Traditional Extraction (without OpenCV)
```bash
# Traditional extraction to JSON
python main.py document.pdf traditional
```

## Performance Optimization

### Content-Aware Processing
- OpenCV pre-analysis reduces unnecessary extraction attempts
- Only relevant extractors are called based on detected content
- Significantly faster for PDFs with specific content types

### Parallel Processing
- Table, image, and text extraction can run in parallel
- Background tasks handle large files without blocking API

### Memory Management
- Temporary files are automatically cleaned after processing
- Page-by-page processing prevents memory overflow

## Output Format

### JSON Structure
```json
{
  "page_1": [
    {
      "type": "text",
      "content": "Document Title",
      "page_number": 1,
      "paragraph_number": 1
    },
    {
      "type": "table",
      "content": [
        ["Header 1", "Header 2"],
        ["Data 1", "Data 2"]
      ],
      "page_number": 1,
      "table_number": 1
    },
    {
      "type": "image",
      "ocr_text": "Chart showing growth",
      "page_number": 1,
      "image_number": 1
    }
  ]
}
```

### Excel Output
- Tables are exported to separate sheets
- Currency formatting preserved (₹ values)
- Headers automatically detected
- Merged cells handled appropriately

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Ensure Tesseract, Ghostscript, and Java are installed
   - Verify PATH environment variables are set correctly

2. **Memory Issues with Large PDFs**
   - Use the asynchronous `/extract/` endpoint
   - Consider processing page ranges separately

3. **Table Detection Issues**
   - Try the CV-enhanced `/extract-tables-to-excel/` endpoint
   - Adjust OpenCV detection parameters in `main.py`

4. **OCR Quality**
   - Ensure Tesseract language packs are installed
   - Higher resolution PDFs produce better OCR results

## License

This project is provided as-is for educational and commercial use.
