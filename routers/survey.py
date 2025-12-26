from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Header
from sqlalchemy.orm import Session
from database import get_db
from schemas import SurveyRegisterRequest, SurveyRegisterResponse, FileUploadResponse
import crud
from typing import List, Optional
import os
import uuid

router = APIRouter()

@router.post("/surveys/register", response_model=SurveyRegisterResponse)
def register_survey(
    request: SurveyRegisterRequest,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Header(..., alias="user-id")
):
    """
    Register a new survey with captions
    """
    try:
        # Convert user_id to integer if provided
        user_id_int = None
        if user_id:
            try:
                user_id_int = int(user_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid user-id format. Must be a number.")

        # Create survey data
        survey_data = {
            "imageUrl": request.imageFile,  # Store filename as imageUrl
            "country": request.country,
            "category": request.category,
            "title": request.title,
            "userId": user_id_int  # Use user ID from header
        }

        # Create survey
        survey = crud.create_survey(db, survey_data)

        # Create captions with type "custom"
        caption_ids = []
        for caption_text in request.caption:
            caption_data = {
                "surveyId": survey.surveyId,
                "text": caption_text,
                "type": "custom"
            }
            caption = crud.create_caption(db, caption_data)
            caption_ids.append(caption.captionId)

        return SurveyRegisterResponse(
            success=True,
            surveyId=survey.surveyId,
            captionIds=caption_ids,
            message="Survey registered successfully"
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to register survey: {str(e)}")

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file and save it to the survey directory
    """
    try:
        # Validate file type (allow webp and other image formats)
        allowed_extensions = {'.webp', '.jpg', '.jpeg', '.png', '.gif'}

        # Define the upload directory
        upload_dir = "/home/teom142/goinfre/culture/web/frontend/LMM/frontend/public/survey"

        # Ensure directory exists
        os.makedirs(upload_dir, exist_ok=True)

        # Validate filename exists
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Get file extension and validate
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"File type {file_extension} not allowed. Allowed types: {', '.join(allowed_extensions)}")

        # Use original filename (already random)
        original_filename = file.filename
        file_path = os.path.join(upload_dir, original_filename)

        # Read and save the uploaded file
        content = await file.read()

        # Validate file content is not empty
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Save the file
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # Verify file was saved correctly
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="File was not saved properly")

        return FileUploadResponse(
            success=True,
            filename=original_filename,
            original_filename=original_filename,
            file_size=len(content),
            message=f"File uploaded successfully to {file_path}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@router.get("/upload/test")
async def test_upload_directory():
    """
    Test endpoint to check upload directory status
    """
    upload_dir = "/home/teom142/goinfre/culture/web/frontend/LMM/frontend/public/survey"

    try:
        # Check if directory exists
        exists = os.path.exists(upload_dir)
        is_dir = os.path.isdir(upload_dir)
        is_writable = os.access(upload_dir, os.W_OK) if exists else False

        # Get directory size and file count
        file_count = 0
        if exists and is_dir:
            file_count = len([f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))])

        return {
            "upload_directory": upload_dir,
            "exists": exists,
            "is_directory": is_dir,
            "is_writable": is_writable,
            "file_count": file_count,
            "status": "ready" if (exists and is_dir and is_writable) else "not ready"
        }
    except Exception as e:
        return {
            "upload_directory": upload_dir,
            "error": str(e),
            "status": "error"
        }