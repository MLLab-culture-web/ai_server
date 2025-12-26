
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
import crud, schemas
from database import get_db

router = APIRouter()

@router.get("/learning-data/", response_model=List[schemas.LearningDataResponse])
def read_learning_data(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    learning_data = crud.get_learning_data(db, skip=skip, limit=limit)
    return learning_data
