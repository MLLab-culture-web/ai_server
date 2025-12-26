
from pydantic import BaseModel
from typing import List, Optional

# Base Schemas
class UserBase(BaseModel):
    username: str

class SurveyBase(BaseModel):
    imageUrl: str
    country: str
    category: str
    title: str

class CaptionBase(BaseModel):
    text: str
    type: str

class ResponseBase(BaseModel):
    cultural: int
    visual: int
    hallucination: int
    time: float

# Schemas for creation
class UserCreate(UserBase):
    password: str

# Schemas for reading
class User(UserBase):
    userId: int

    class Config:
        orm_mode = True

class Caption(CaptionBase):
    captionId: int

    class Config:
        orm_mode = True

class Survey(SurveyBase):
    surveyId: int
    captions: List[Caption] = []

    class Config:
        orm_mode = True

class Response(ResponseBase):
    responseId: int
    user: User

    class Config:
        orm_mode = True

# Schema for learning data export
class LearningDataResponse(BaseModel):
    responseId: int
    title: str
    category: str
    country: str
    cultural: int
    visual: int
    hallucination: int
    time: float
    user: User
    caption: Caption
    

    class Config:
        orm_mode = True

# Schemas for evaluation
class EvaluationRequest(BaseModel):
    startCaptionId: int
    endCaptionId: int

class EvaluationRequestWithFlag(BaseModel):
    startCaptionId: int
    endCaptionId: int
    flag: int

class EvaluationResponse(BaseModel):
    start_caption_id: int
    end_caption_id: int
    total_captions: int
    evaluated_successfully: int
    failed_evaluations: int
    results: List[dict]  # 각 캡션별 결과
    summary: dict
    error_samples: Optional[List[dict]] = []

class EvaluateAllWithFlagRequest(BaseModel):
    start_flag: Optional[int] = 50  # Default starts from flag 50

class EvaluateAllResponse(BaseModel):
    total_captions: int
    evaluated_successfully: int
    failed_evaluations: int
    deleted_previous_evaluations: int
    summary: dict
    error_samples: Optional[List[dict]] = []

class RefreshMetricResponse(BaseModel):
    total_caption_pairs: int
    processed_successfully: int
    failed_calculations: int
    deleted_previous_metrics: int
    summary: dict
    error_samples: Optional[List[dict]] = []

# Survey registration schemas
class SurveyRegisterRequest(BaseModel):
    imageFile: str
    country: str
    category: str
    title: str
    caption: List[str]

class SurveyRegisterResponse(BaseModel):
    success: bool
    surveyId: int
    captionIds: List[int]
    message: str

class FileUploadResponse(BaseModel):
    success: bool
    filename: str
    original_filename: str
    file_size: int
    message: str

class ImageFreshResponse(BaseModel):
    success: bool
    processed_surveys: int
    updated_surveys: int
    updated_list: List[dict]
    message: str

# Response count schemas
class SurveyResponseCount(BaseModel):
    surveyId: int
    title: str
    country: str
    category: str
    response_count: int

    class Config:
        orm_mode = True

class ResponseCountResponse(BaseModel):
    total_surveys: int
    survey_counts: List[SurveyResponseCount]

# Unseen evaluation schemas
class UnseenEvaluationRequest(BaseModel):
    survey_titles: List[str]

class UnseenEvaluationResponse(BaseModel):
    total_surveys: int
    total_captions: int
    evaluated_successfully: int
    failed_evaluations: int
    deleted_previous_evaluations: int
    summary: dict
    error_samples: Optional[List[dict]] = []

# Variance analysis schemas
class CategoryWassersteinStats(BaseModel):
    category: str
    mean: float
    variance: float

class UserResponseStats(BaseModel):
    mean_responses_per_user: float
    variance_responses_per_user: float

class CategoryResponseDistribution(BaseModel):
    category: str
    response_count: int

class VarianceAnalysisResponse(BaseModel):
    wasserstein_by_category: List[CategoryWassersteinStats]
    user_response_stats: UserResponseStats
    response_distribution_by_category: List[CategoryResponseDistribution]

