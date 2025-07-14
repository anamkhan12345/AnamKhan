from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import shutil
import os
import sqlite3
from datetime import datetime
#from cloud_utils import upload_to_gcs

app = FastAPI()

@app.post("/upload")
async def upload_detection(
    image: UploadFile = File(...),
    label: str = Form(...),
    confidence: float = Form(...),
    timestamp: str = Form(...)
):
    # Step 1: Save image locally
    #safe_filename = f"{timestamp.replace(' ', '_')}_{image.filename}"
    #local_path = os.path.join("images", safe_filename)
    #os.makedirs("images", exist_ok=True)
    
    #with open(local_path, "wb") as buffer:
    #    shutil.copyfileobj(image.file, buffer)

    # Step 2: Upload image to GCS
    #gcs_url = upload_to_gcs(BUCKET_NAME, local_path, f"detections/{safe_filename}")

    # Step 3: Write metadata to SQLite
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO detections (filename, label, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    """, (image.filename, label, confidence, timestamp) )
    conn.commit()
    conn.close()

    # Step 4: Return response
    metadata = {
        "filename": image.filename,
        "label": label,
        "confidence": confidence,
        "timestamp": timestamp,
    }

    return JSONResponse(content=metadata)
