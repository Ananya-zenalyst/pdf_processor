from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
import json
import uuid

# Import the main extraction function from our command-line script
from main import analyze_and_extract

# Initialize the FastAPI app
app = FastAPI(
    title="PDF Data Extraction API",
    description="Upload a PDF and get structured text, table, and image data.",
    version="1.0.0"
)

# Define the directory to store uploaded files
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

@app.post("/extract/")
async def create_upload_file(file: UploadFile = File(...)):
    """
    Accepts a PDF file, saves it temporarily, processes it to extract
    data, and returns the data as JSON.
    """
    # Generate a unique filename to avoid conflicts
    unique_filename = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOADS_DIR, unique_filename)

    try:
        # Save the uploaded file to the server
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Call our existing, powerful extraction logic
        print(f"Processing file: {file_path}")
        extracted_data_json = analyze_and_extract(file_path)
        
        # The function returns a JSON string, so we parse it back to a dict
        extracted_data = json.loads(extracted_data_json)

        # Check if there was an error during extraction
        if "error" in extracted_data:
            raise HTTPException(status_code=500, detail=extracted_data["error"])

        return JSONResponse(content=extracted_data)

    except Exception as e:
        # Catch any other exceptions
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    finally:
        # Clean up: remove the uploaded file after processing
        if os.path.exists(file_path):
            os.remove(file_path)
        # Also clean up any extracted images if the folder exists
        image_output_dir = os.path.join(UPLOADS_DIR, 'extracted_images')
        if os.path.exists(image_output_dir):
            shutil.rmtree(image_output_dir)

@app.get("/")
def read_root():
    """
    Root endpoint with a welcome message and instructions.
    """
    return {"message": "Welcome to the PDF Extraction API. Please go to /docs to see the API documentation and upload a file."}
