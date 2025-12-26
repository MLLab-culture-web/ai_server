from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from admin import admin_app  # Import from our new admin.py
from database import engine
import models
from routers import admin as admin_router, data as data_router, evaluation as evaluation_router, survey as survey_router, evaluation_unseen as evaluation_unseen_router # Import routers
from starlette.responses import RedirectResponse

# Create all tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="CultureLens Admin Dashboard",
    description="An admin dashboard to manage and view data for the CultureLens project.",
    version="1.0.0",
)

# Include routers
app.include_router(admin_router.router, prefix="/admin", tags=["admin-api"])
app.include_router(data_router.router, prefix="/admin", tags=["data-api"])
app.include_router(evaluation_router.router, prefix="/api", tags=["evaluation-api"])
app.include_router(evaluation_unseen_router.router, prefix="/api", tags=["evaluation-unseen-api"])
app.include_router(survey_router.router, prefix="/api", tags=["survey-api"])


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://culturelens.ngrok.io", "http://localhost:3000"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the admin app
admin_app.mount_to(app)

@app.get("/")
def read_root():
    return RedirectResponse(url="/admin/user/list")