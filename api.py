import os
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.responses import StreamingResponse
import io

# Import the main extraction function from your controller
from main import get_document_layout
from extractors.table_extractor import extract_tables_to_excel, convert_main_layout_to_excel

app = FastAPI(
    title="PDF Data Extraction API",
    description="An advanced API to extract text, tables, and images from a PDF, preserving the document's spatial layout.",
    version="3.0.0"
)

# In-memory "database" to store task status and results.
# In a production environment, you would replace this with a real database like Redis.
tasks_db = {}

# Ensure the directory for uploads exists
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)


def process_pdf_in_background(task_id: str, pdf_path: str):
    """
    This function runs in the background to process the PDF.
    It updates the task status in the tasks_db as it progresses.
    """
    try:
        # Update status to PROCESSING
        tasks_db[task_id]["status"] = "PROCESSING"
        
        # Call the core extraction logic from main.py
        structured_data = get_document_layout(pdf_path)
        
        # On success, update status and store the result
        tasks_db[task_id]["status"] = "SUCCESS"
        tasks_db[task_id]["result"] = structured_data

    except Exception as e:
        # On failure, update status and store the error message
        print(f"Error processing task {task_id}: {e}")
        tasks_db[task_id]["status"] = "FAILURE"
        tasks_db[task_id]["result"] = {"error": str(e)}
    finally:
        # Clean up by removing the uploaded file after processing
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


@app.post("/extract/", status_code=202)
async def extract_data_from_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a PDF file upload, starts a background process to extract data,
    and immediately returns a task ID for status polling.
    """
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Define the path where the uploaded file will be saved
    file_path = os.path.join(UPLOADS_DIR, f"{task_id}_{file.filename}")

    # Save the uploaded file
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Initialize the task in our "database"
    tasks_db[task_id] = {"status": "PENDING", "result": None}

    # Add the processing function to run in the background
    background_tasks.add_task(process_pdf_in_background, task_id, file_path)

    # Return the task ID to the client
    return {"task_id": task_id, "message": "File upload successful. Processing has started."}


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Poll this endpoint with a task ID to check the status of the extraction
    process and retrieve the final JSON result when complete.
    """
    task = tasks_db.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    if task["status"] == "SUCCESS":
        return JSONResponse(content={"status": task["status"], "data": task["result"]})
    
    if task["status"] == "FAILURE":
        return JSONResponse(status_code=500, content={"status": task["status"], "error": task["result"]})
        
    return {"status": task["status"]}

@app.post("/extract-to-excel/")
async def extract_pdf_to_excel(file: UploadFile = File(...)):
    """
    Extract ONLY TABLE DATA from PDF and return as Excel file download.

    This endpoint uses the main extraction pipeline but converts ONLY tables
    and table-like text content to Excel format. Non-table text (headers,
    signatures, etc.) is excluded.

    Features:
    - Detects structured tables if available
    - Identifies text that contains table patterns (S.No, Qty, Rate, Amount, etc.)
    - Parses text-based tables into proper Excel format
    - Applies currency formatting for ₹ values
    - Excludes non-tabular content
    """
    # Generate a unique filename for the uploaded file
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOADS_DIR, f"{file_id}_{file.filename}")

    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Use the main extraction pipeline
        layout_data = get_document_layout(file_path)

        # Convert to Excel using the main layout data
        excel_bytes = convert_main_layout_to_excel(layout_data)

        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

        # Create filename for download
        base_filename = os.path.splitext(file.filename)[0] if file.filename else "extracted_content"
        excel_filename = f"{base_filename}_content.xlsx"

        # Return Excel file as download
        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={excel_filename}"}
        )

    except Exception as e:
        # Clean up file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract content and convert to Excel: {str(e)}"
        )

@app.post("/extract-tables-to-excel/")
async def extract_only_tables_to_excel(file: UploadFile = File(...)):
    """
    Extract only tables from PDF using CV-enhanced method and return as Excel file.

    This endpoint uses the advanced computer vision table extraction
    and may find different tables than the main extraction pipeline.
    """
    # Generate a unique filename for the uploaded file
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOADS_DIR, f"{file_id}_{file.filename}")

    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extract tables using CV-enhanced method
        excel_bytes = extract_tables_to_excel(file_path)

        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

        # Create filename for download
        base_filename = os.path.splitext(file.filename)[0] if file.filename else "extracted_tables"
        excel_filename = f"{base_filename}_tables_cv.xlsx"

        # Return Excel file as download
        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={excel_filename}"}
        )

    except Exception as e:
        # Clean up file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract tables with CV method and convert to Excel: {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "Welcome to the PDF Data Extraction API. Go to /docs to see the API documentation."}

