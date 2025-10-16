Advanced PDF Data Extraction API
This project provides a powerful, multi-stage API for extracting text, tables, and images from PDF documents. It is designed to preserve the spatial layout of the original document and uses a combination of pdfplumber, camelot, and opencv for state-of-the-art table extraction.

Features
Spatially-Aware Extraction: Reconstructs the top-to-bottom, left-to-right reading order of the PDF.

Intelligent Text Grouping: Correctly groups text into paragraphs, even in multi-column layouts.

Advanced Table Extraction: A three-stage pipeline for maximum accuracy:

Camelot: For tables with clear borders.

pdfplumber: For borderless tables based on text alignment.

OpenCV: A computer vision fallback for difficult or scanned tables.

Chart Data Extraction: Pulls both digital text and OCR'd text from charts and images.

Content De-duplication: Intelligently removes redundant text that is already part of an extracted table or chart.

Asynchronous Processing: Handles large files efficiently with background tasks.

Prerequisites
Before you begin, you must have the following external software installed on your system:

Python 3.8+: Download Python

Tesseract OCR: Required for reading text from images.

Installation Guide for Tesseract

Ghostscript: A required dependency for camelot.

Download Ghostscript

Java Runtime Environment: A required dependency for tabula-py, which is used within our advanced table extractor.

Download Java

Setup
Create Project Structure:
Create the main folder data_extraction_api and the sub-folders extractors and uploads. Place all the Python files (.py) in their respective locations.

Create a Virtual Environment (Recommended):
Navigate to the data_extraction_api directory in your terminal and run:

python3 -m venv venv

Activate the Virtual Environment:

On macOS/Linux: source venv/bin/activate

On Windows: venv\Scripts\activate

Install Dependencies:
With your virtual environment active, install all the required Python libraries:

pip install -r requirements.txt

How to Run the API
Start the Server:
From the root data_extraction_api directory, run the following command in your terminal:

uvicorn api:app --reload

The server will start, typically on http://127.0.0.1:8000.

Access the API Docs:
Open your web browser and navigate to http://127.0.0.1:8000/docs. This will open the interactive FastAPI documentation where you can test the API.

API Usage
The API uses a background task model to handle PDF processing.

1. Upload a PDF
Go to the /extract/ endpoint in the API docs.

Click "Try it out" and select a PDF file to upload.

Execute the request. You will receive an immediate response with a unique task_id.

Example Response:

{
  "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "message": "File upload successful. Processing has started."
}

2. Check the Status
Copy the task_id from the previous step.

Go to the /status/{task_id} endpoint in the API docs.

Click "Try it out", paste the task_id, and execute.

The status will be PENDING, PROCESSING, SUCCESS, or FAILURE.

3. Retrieve the Results
Keep polling the /status/{task_id} endpoint until the status changes to SUCCESS.

When successful, the response will contain the full, spatially-ordered JSON data.

Example Success Response:

{
  "status": "SUCCESS",
  "data": {
    "page_1": [
      {
        "type": "text",
        "content": "Report Title",
        "bbox": [100, 50, 300, 70]
      },
      {
        "type": "table",
        "content": { ... table data ... },
        "bbox": [100, 100, 500, 300]
      },
      {
        "type": "text",
        "content": "A paragraph explaining the table.",
        "bbox": [100, 320, 500, 350]
      }
    ]
  }
}
