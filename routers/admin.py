
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
import crud, schemas
from database import get_db
import re

router = APIRouter()

@router.get("/users/", response_model=List[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users

@router.get("/surveys/", response_model=List[schemas.Survey])
def read_surveys(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    surveys = crud.get_surveys(db, skip=skip, limit=limit)
    return surveys

@router.get("/captions/", response_model=List[schemas.Caption])
def read_captions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    captions = crud.get_captions(db, skip=skip, limit=limit)
    return captions

@router.get("/responses/", response_model=List[schemas.Response])
def read_responses(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    responses = crud.get_responses(db, skip=skip, limit=limit)
    return responses

@router.get("/responses/count", response_model=schemas.ResponseCountResponse)
def get_response_counts(db: Session = Depends(get_db)):
    """
    Get response count for each survey, ordered by count descending
    """
    try:
        results = crud.get_response_counts_by_survey(db)

        survey_counts = []
        for survey, response_count in results:
            survey_counts.append(schemas.SurveyResponseCount(
                surveyId=survey.surveyId,
                title=survey.title,
                country=survey.country,
                category=survey.category,
                response_count=response_count
            ))

        return schemas.ResponseCountResponse(
            total_surveys=len(survey_counts),
            survey_counts=survey_counts
        )

    except Exception:
        return schemas.ResponseCountResponse(
            total_surveys=0,
            survey_counts=[]
        )

@router.get("/image-fresh", response_model=schemas.ImageFreshResponse)
def refresh_image_urls(db: Session = Depends(get_db)):
    """
    Update survey imageUrls from jpg format to webp format
    Changes URLs like:
    https://culturelens.cloud/upload/survey/517aa42d-6505-4140-8c4f-755fd4b6631d-tmp2w5e8qkm.jpg
    to:
    517aa42d-6505-4140-8c4f-755fd4b6631d-tmp2w5e8qkm.webp
    """
    try:
        # Get all surveys
        surveys = crud.get_all_surveys(db)

        updated_count = 0
        processed_count = 0
        updated_surveys = []

        # Pattern to match the URL format and extract filename
        url_pattern = r'https://culturelens\.cloud/upload/survey/(.+)\.jpg$'

        for survey in surveys:
            processed_count += 1

            if survey.imageUrl:
                match = re.match(url_pattern, survey.imageUrl)
                if match:
                    # Extract filename without extension
                    filename_base = match.group(1)

                    # Create new webp filename
                    new_image_url = f"{filename_base}.webp"

                    # Update the survey
                    crud.update_survey_image_url(db, survey.surveyId, new_image_url)

                    updated_count += 1
                    updated_surveys.append({
                        "surveyId": survey.surveyId,
                        "title": survey.title,
                        "old_url": survey.imageUrl,
                        "new_url": new_image_url
                    })

        return schemas.ImageFreshResponse(
            success=True,
            processed_surveys=processed_count,
            updated_surveys=updated_count,
            updated_list=updated_surveys,
            message=f"Successfully processed {processed_count} surveys and updated {updated_count} image URLs"
        )

    except Exception as e:
        return schemas.ImageFreshResponse(
            success=False,
            processed_surveys=0,
            updated_surveys=0,
            updated_list=[],
            message=f"Failed to refresh image URLs: {str(e)}"
        )

