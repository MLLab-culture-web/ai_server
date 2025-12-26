from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import crud, schemas, models
from database import get_db
from .evaluation import (
    eval_logger,
    _download_image_as_base64,
    find_latest_eval_metrics_file,
    find_similar_captions_with_responses_json,
    create_few_shot_examples,
    _evaluate_metric
)

router = APIRouter()

@router.post("/evaluate-unseen", response_model=schemas.UnseenEvaluationResponse)
def evaluate_unseen_surveys(
    request: schemas.UnseenEvaluationRequest,
    db: Session = Depends(get_db)
):
    """
    Evaluate specific surveys treating them as if they have no responses
    Saves results to AgentEvalDetailUnseen table with flag increments of 50
    """
    eval_logger.info(f"🚀 Starting evaluate-unseen for surveys: {request.survey_titles}")

    # Get captions for the specified survey titles
    target_captions = crud.get_captions_by_survey_titles(db, request.survey_titles)
    total_captions = len(target_captions)

    if not target_captions:
        eval_logger.warning("⚠️ No captions found for the specified survey titles")
        return schemas.UnseenEvaluationResponse(
            total_surveys=len(request.survey_titles),
            total_captions=0,
            evaluated_successfully=0,
            failed_evaluations=0,
            deleted_previous_evaluations=0,
            summary={"message": "No captions found for specified survey titles"},
            error_samples=[]
        )

    eval_logger.info(f"📊 Found {total_captions} captions across {len(request.survey_titles)} surveys")

    # Delete existing unseen evaluations for these captions
    caption_ids = [caption.captionId for caption in target_captions]
    deleted_count = crud.delete_agent_eval_detail_unseen_by_caption_ids(db, caption_ids)
    eval_logger.info(f"🗑️ Deleted {deleted_count} existing unseen evaluation records")

    evaluated_successfully = 0
    failed_evaluations = 0
    error_samples = []

    # Get all responses ordered by ID to calculate actual data-based thresholds
    all_responses = db.query(models.Response).order_by(models.Response.responseId.asc()).all()
    total_response_count = len(all_responses)
    eval_logger.info(f"📊 Total responses in database: {total_response_count}")

    # Calculate flag values based on actual 50 data increments
    flag_values = []
    current_flag = 50
    response_index = 50  # First flag at 50 responses

    while response_index <= total_response_count:
        # Get the response ID at this index (50th, 100th, 150th... response)
        response_threshold_id = all_responses[response_index - 1].responseId
        flag_values.append((response_threshold_id, current_flag))
        response_index += 50
        current_flag += 50

    eval_logger.info(f"📊 Flag values to generate: {flag_values}")

    if not flag_values:
        eval_logger.warning("⚠️ No flag values to generate (max_response_id < 50)")
        return schemas.UnseenEvaluationResponse(
            total_surveys=len(request.survey_titles),
            total_captions=total_captions,
            evaluated_successfully=0,
            failed_evaluations=0,
            deleted_previous_evaluations=deleted_count,
            summary={"message": "No response data available for flag generation"},
            error_samples=[]
        )

    eval_logger.info(f"🚀 Starting evaluation of {total_captions} unseen captions with {len(flag_values)} flag configurations...")

    for flag_threshold, flag_value in flag_values:
        eval_logger.info(f"🏷️ === Processing Flag {flag_value} (Response threshold: {flag_threshold}) ===")

        # Process each caption for this flag
        for idx, caption in enumerate(target_captions, 1):
            try:
                # Skip captions without survey or imageUrl
                if not caption.survey or not caption.survey.imageUrl:
                    eval_logger.error(f"Image URL not found for caption {caption.captionId}")
                    failed_evaluations += 1
                    continue

                # Download and encode image
                try:
                    image_base64 = _download_image_as_base64(caption.survey.imageUrl)
                except HTTPException as img_error:
                    eval_logger.error(f"Image download failed for caption {caption.captionId}: {img_error.detail}")
                    failed_evaluations += 1
                    continue

                # For unseen evaluation, we treat as if there are no responses for this caption
                # So we skip the current_distributions logic entirely
                eval_logger.info(f"📊 Treating caption {caption.captionId} as unseen (no response data used)")

                # Find similar captions for few-shot examples using latest JSON metrics
                latest_json_file = find_latest_eval_metrics_file()
                similar_captions_with_responses = find_similar_captions_with_responses_json(
                    db, caption.text, image_base64, caption.captionId, caption.surveyId,
                    latest_json_file, top_k=5
                )

                # Create few-shot examples for each metric (자기 자신 제외)
                cultural_examples = create_few_shot_examples(
                    similar_captions_with_responses, 'cultural'
                )
                visual_examples = create_few_shot_examples(
                    similar_captions_with_responses, 'visual'
                )
                hallucination_examples = create_few_shot_examples(
                    similar_captions_with_responses, 'hallucination'
                )

                # Prepare artifact information
                artifact_info = {
                    'title': caption.survey.title if caption.survey else None,
                    'country': caption.survey.country if caption.survey else None,
                    'category': caption.survey.category if caption.survey else None
                }

                # Evaluate each metric
                cultural_result, cultural_debug = _evaluate_metric(caption.text, image_base64, 'cultural', cultural_examples, artifact_info)
                visual_result, visual_debug = _evaluate_metric(caption.text, image_base64, 'visual', visual_examples, artifact_info)
                hallucination_result, hallucination_debug = _evaluate_metric(caption.text, image_base64, 'hallucination', hallucination_examples, artifact_info)

                # Save to AgentEvalDetailUnseen table
                try:
                    created_details = crud.create_agent_eval_detail_unseen_samples(
                        db=db,
                        caption_id=caption.captionId,
                        cultural_dist=cultural_result,
                        visual_dist=visual_result,
                        hallucination_dist=hallucination_result,
                        flag=flag_value
                    )
                    eval_logger.info(f"✅ Saved {len(created_details)} unseen evaluation details for caption {caption.captionId} with flag {flag_value}")

                except Exception as db_error:
                    eval_logger.error(f"Failed to save unseen evaluation details for caption {caption.captionId} with flag {flag_value}: {db_error}")
                    failed_evaluations += 1
                    continue

                eval_logger.info(f"✅ Caption {caption.captionId} evaluated successfully with flag {flag_value}! ({idx}/{total_captions} captions for this flag)")

            except Exception as e:
                eval_logger.error(f"Failed to evaluate caption {caption.captionId} with flag {flag_value}: {e}")
                failed_evaluations += 1

                if len(error_samples) < 5:
                    error_samples.append({
                        "caption_id": caption.captionId,
                        "flag": flag_value,
                        "error": str(e),
                        "error_type": type(e).__name__
                    })

        eval_logger.info(f"✅ Completed processing flag {flag_value} for all {total_captions} unseen captions")

    # Calculate success count
    total_operations = total_captions * len(flag_values)
    evaluated_successfully = total_operations - failed_evaluations

    # Prepare summary
    summary = {
        "survey_titles": request.survey_titles,
        "total_captions": total_captions,
        "flag_configurations": len(flag_values),
        "flag_values_generated": [flag for _, flag in flag_values],
        "total_operations": total_operations,
        "max_response_id_used": max_response_id,
        "response_thresholds": [threshold for threshold, _ in flag_values],
        "data_structure": "AgentEvalDetailUnseen"
    }

    eval_logger.info(f"🏁 === Evaluate-unseen Complete ===")
    eval_logger.info(f"📊 Target surveys: {request.survey_titles}")
    eval_logger.info(f"📊 Total captions: {total_captions}")
    eval_logger.info(f"📊 Flag configurations: {len(flag_values)}")
    eval_logger.info(f"📊 Total operations: {total_operations}")
    eval_logger.info(f"✅ Successfully completed: {evaluated_successfully}")
    eval_logger.info(f"❌ Failed operations: {failed_evaluations}")

    return schemas.UnseenEvaluationResponse(
        total_surveys=len(request.survey_titles),
        total_captions=total_captions,
        evaluated_successfully=evaluated_successfully,
        failed_evaluations=failed_evaluations,
        deleted_previous_evaluations=deleted_count,
        summary=summary,
        error_samples=error_samples
    )