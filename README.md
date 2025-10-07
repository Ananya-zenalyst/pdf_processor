PDF Data Extraction API (Asynchronous)
This project is a Python-based API for extracting text, tables, and image data from PDF files. It uses a smart controller to analyze PDFs and a background processing model to handle large files efficiently.

Features
Intelligent Analysis: Automatically detects text, tables, and images to run only necessary extractors.

Modular Extractors: Separate, best-in-class tools for text, tables, and OCR on images.

Asynchronous Processing: Handles large files without timeouts by processing them in the background.

Simple API: Easy to use with a clear, interactive documentation page.

Setup
Install Python Libraries:
Navigate to the data_extraction_api folder and run:

pip install -r requirements.txt

Install Tesseract OCR:
This is required for reading text from images. Find instructions for your OS here: https://github.com/tesseract-ocr/tesseract

How to Run the Web API
Start the Server:
From the main data_extraction_api folder, run the following command:

uvicorn api:app --reload

Use the API:

Open your browser to https://www.google.com/search?q=http://127.0.0.1:8000/docs.

Use the POST /extract/ endpoint to upload your PDF. You will get a task_id back immediately.

Use the GET /status/{task_id} endpoint with your task_id to check the progress and retrieve the final JSON data once processing is complete.