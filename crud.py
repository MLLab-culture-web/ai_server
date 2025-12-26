
from sqlalchemy.orm import Session, joinedload
import models, schemas

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

def get_surveys(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Survey).options(joinedload(models.Survey.captions)).offset(skip).limit(limit).all()

def get_captions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Caption).offset(skip).limit(limit).all()

def get_caption_by_id(db: Session, caption_id: int):
    return (
        db.query(models.Caption)
        .options(joinedload(models.Caption.survey))
        .filter(models.Caption.captionId == caption_id)
        .first()
    )

def get_responses(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Response).options(joinedload(models.Response.user)).offset(skip).limit(limit).all()

def get_learning_data(db: Session, skip: int = 0, limit: int = 100):
    results = (
        db.query(models.Response)
        .options(
            joinedload(models.Response.user),
            joinedload(models.Response.caption).joinedload(models.Caption.survey)
        )
        .offset(skip)
        .limit(limit)
        .all()
    )
    
    response_data = []
    for res in results:
        response_data.append(
            schemas.LearningDataResponse(
                responseId=res.responseId,
                title=res.survey_title,
                category=res.caption.survey.category,
                country=res.caption.survey.country,
                cultural=res.cultural,
                visual=res.visual,
                hallucination=res.hallucination,
                time=res.time,
                user=res.user,
                caption=res.caption,
            )
        )
    return response_data


def get_all_captions_with_survey(db: Session):
    """Get all captions with their survey information"""
    return (
        db.query(models.Caption)
        .options(joinedload(models.Caption.survey))
        .all()
    )


def delete_all_eval_metrics(db: Session):
    """Delete all evaluation metrics"""
    try:
        deleted_count = db.query(models.EvalMetric).delete()
        db.commit()
        return deleted_count
    except Exception as e:
        db.rollback()
        raise e

def create_eval_metric(db: Session, source_caption_id: int, target_caption_id: int, caption_sim: float, image_sim: float):
    """Create a new evaluation metric record"""
    try:
        db_metric = models.EvalMetric(
            sourceCaptionId=source_caption_id,
            targetCaptionId=target_caption_id,
            captionSim=caption_sim,
            imageSim=image_sim
        )
        db.add(db_metric)
        db.commit()
        db.refresh(db_metric)
        return db_metric
    except Exception as e:
        db.rollback()
        raise e

def bulk_create_eval_metrics(db: Session, metrics: list):
    """Bulk create evaluation metric records"""
    try:
        db_metrics = []
        for metric in metrics:
            db_metric = models.EvalMetric(
                sourceCaptionId=metric['source_caption_id'],
                targetCaptionId=metric['target_caption_id'],
                captionSim=metric['caption_sim'],
                imageSim=metric['image_sim']
            )
            db_metrics.append(db_metric)

        db.add_all(db_metrics)
        db.commit()
        return db_metrics
    except Exception as e:
        db.rollback()
        raise e

def get_eval_metrics_by_source_caption(db: Session, source_caption_id: int, exclude_survey_id: int = None):
    """Get all evaluation metrics for a source caption, optionally excluding a survey"""
    query = (
        db.query(models.EvalMetric)
        .join(models.Caption, models.EvalMetric.targetCaptionId == models.Caption.captionId)
        .filter(models.EvalMetric.sourceCaptionId == source_caption_id)
    )

    if exclude_survey_id:
        query = query.filter(models.Caption.surveyId != exclude_survey_id)

    return query.all()

def get_eval_metrics_count(db: Session):
    """Get total count of evaluation metrics"""
    return db.query(models.EvalMetric).count()

def check_eval_metric_exists(db: Session, source_caption_id: int, target_caption_id: int):
    """Check if evaluation metric already exists for a source-target pair"""
    return db.query(models.EvalMetric).filter(
        models.EvalMetric.sourceCaptionId == source_caption_id,
        models.EvalMetric.targetCaptionId == target_caption_id
    ).first() is not None

def get_existing_eval_metric_pairs(db: Session):
    """Get all existing source-target caption pairs as a set for fast lookup"""
    pairs = db.query(
        models.EvalMetric.sourceCaptionId,
        models.EvalMetric.targetCaptionId
    ).all()
    return set((pair.sourceCaptionId, pair.targetCaptionId) for pair in pairs)

def remove_symmetric_duplicates(db: Session):
    """
    Remove symmetric duplicate pairs from eval metrics.
    Keep only pairs where sourceCaptionId < targetCaptionId
    """
    try:
        # Get all eval metrics with their IDs
        all_metrics = db.query(models.EvalMetric).all()

        # Group metrics by symmetric pairs
        pair_groups = {}
        for metric in all_metrics:
            source_id = metric.sourceCaptionId
            target_id = metric.targetCaptionId

            # Create a canonical key (smaller ID first)
            if source_id < target_id:
                key = (source_id, target_id)
            else:
                key = (target_id, source_id)

            if key not in pair_groups:
                pair_groups[key] = []
            pair_groups[key].append(metric)

        # Find duplicates to delete
        ids_to_delete = []
        kept_count = 0

        for (canonical_source, canonical_target), metrics in pair_groups.items():
            if len(metrics) > 1:
                # Multiple records for the same symmetric pair
                # Keep the one where sourceCaptionId < targetCaptionId
                keep_metric = None
                delete_metrics = []

                for metric in metrics:
                    if metric.sourceCaptionId == canonical_source and metric.targetCaptionId == canonical_target:
                        # This is the canonical form (source < target)
                        if keep_metric is None:
                            keep_metric = metric
                        else:
                            # Multiple canonical forms, keep the first one
                            delete_metrics.append(metric)
                    else:
                        # This is the reverse form (source > target)
                        delete_metrics.append(metric)

                if keep_metric is None and delete_metrics:
                    # No canonical form found, keep the first one and convert it
                    keep_metric = delete_metrics.pop(0)
                    # Update to canonical form
                    keep_metric.sourceCaptionId = canonical_source
                    keep_metric.targetCaptionId = canonical_target

                # Collect IDs to delete
                for metric in delete_metrics:
                    ids_to_delete.append(metric.id)

                if keep_metric:
                    kept_count += 1
            else:
                # Single record, check if it needs to be converted to canonical form
                metric = metrics[0]
                if metric.sourceCaptionId > metric.targetCaptionId:
                    # Convert to canonical form
                    metric.sourceCaptionId, metric.targetCaptionId = metric.targetCaptionId, metric.sourceCaptionId
                kept_count += 1

        # Delete duplicates
        deleted_count = 0
        if ids_to_delete:
            deleted_count = db.query(models.EvalMetric).filter(
                models.EvalMetric.id.in_(ids_to_delete)
            ).delete(synchronize_session=False)

        db.commit()

        return {
            'deleted_count': deleted_count,
            'kept_count': kept_count,
            'total_unique_pairs': len(pair_groups)
        }

    except Exception as e:
        db.rollback()
        raise e

def get_all_eval_metrics_as_json(db: Session):
    """Get all evaluation metrics as JSON-serializable data"""
    try:
        metrics = db.query(models.EvalMetric).all()

        metrics_data = []
        for metric in metrics:
            metrics_data.append({
                "id": metric.id,
                "sourceCaptionId": metric.sourceCaptionId,
                "targetCaptionId": metric.targetCaptionId,
                "captionSim": float(metric.captionSim) if metric.captionSim is not None else None,
                "imageSim": float(metric.imageSim) if metric.imageSim is not None else None,
                "createdAt": metric.createdAt.isoformat() if hasattr(metric, 'createdAt') and metric.createdAt else None
            })

        return {
            "total_count": len(metrics_data),
            "metrics": metrics_data
        }

    except Exception as e:
        raise e

def create_agent_eval_detail_samples(db: Session, caption_id: int, cultural_dist: dict, visual_dist: dict, hallucination_dist: dict, flag: int):
    """
    Create AgentEvalDetail records from distribution data

    Args:
        db: Database session
        caption_id: Caption ID to associate with
        cultural_dist: Cultural distribution {'1': '0%', '2': '0%', '3': '20%', '4': '50%', '5': '30%'}
        visual_dist: Visual distribution
        hallucination_dist: Hallucination distribution
        flag: Flag value from API request

    Returns:
        List of created AgentEvalDetail objects
    """
    print(f"🗄️ CRUD: Starting create_agent_eval_detail_samples for caption {caption_id} with flag {flag}")
    print(f"🗄️ CRUD: Input distributions - cultural: {cultural_dist}, visual: {visual_dist}, hallucination: {hallucination_dist}")

    try:
        # First delete any existing agent eval details for this caption
        # deleted_count = db.query(models.AgentEvalDetail).filter(
        #     models.AgentEvalDetail.captionId == caption_id
        # ).delete()
        # print(f"🗄️ CRUD: Deleted {deleted_count} existing records for caption {caption_id}")

        created_details = []

        # Process each metric type
        distributions = {
            'cultural': cultural_dist,
            'visual': visual_dist,
            'hallucination': hallucination_dist
        }

        print(f"🗄️ CRUD: Processing {len(distributions)} distribution types")

        for eval_type, distribution in distributions.items():
            print(f"🗄️ CRUD: Processing {eval_type} distribution: {distribution}")
            for likert_str, value_str in distribution.items():
                likert = int(likert_str)
                # Convert percentage string to float (remove % and convert)
                value = float(value_str.rstrip('%'))

                print(f"🗄️ CRUD: Creating record - type: {eval_type}, likert: {likert}, value: {value}, flag: {flag}, captionId: {caption_id}")

                db_detail = models.AgentEvalDetail(
                    type=eval_type,
                    likert=likert,
                    value=value,
                    flag=flag,
                    captionId=caption_id
                )
                db.add(db_detail)
                created_details.append(db_detail)

        print(f"🗄️ CRUD: Created {len(created_details)} records, committing to database...")
        db.commit()
        print(f"🗄️ CRUD: Database commit successful")

        # Refresh all created details
        for detail in created_details:
            db.refresh(detail)

        print(f"🗄️ CRUD: Successfully created and refreshed {len(created_details)} AgentEvalDetail records")
        return created_details

    except Exception as e:
        print(f"🗄️ CRUD: Error occurred - {str(e)}")
        db.rollback()
        print(f"🗄️ CRUD: Database rolled back")
        raise e

def delete_all_agent_eval_detail_samples(db: Session):
    """Delete all agent evaluation detail samples"""
    try:
        deleted_count = db.query(models.AgentEvalDetail).delete()
        db.commit()
        return deleted_count
    except Exception as e:
        db.rollback()
        raise e

def get_agent_eval_details_by_caption_id(db: Session, caption_id: int):
    """Get all agent evaluation details for a caption"""
    return db.query(models.AgentEvalDetail).filter(
        models.AgentEvalDetail.captionId == caption_id
    ).all()

def create_survey(db: Session, survey_data: dict):
    """Create a new survey"""
    try:
        db_survey = models.Survey(
            imageUrl=survey_data["imageUrl"],
            country=survey_data["country"],
            category=survey_data["category"],
            title=survey_data["title"],
            userId=survey_data.get("userId")  # Can be None for admin-created surveys
        )
        db.add(db_survey)
        db.commit()
        db.refresh(db_survey)
        return db_survey
    except Exception as e:
        db.rollback()
        raise e

def create_caption(db: Session, caption_data: dict):
    """Create a new caption"""
    try:
        db_caption = models.Caption(
            surveyId=caption_data["surveyId"],
            text=caption_data["text"],
            type=caption_data["type"]
        )
        db.add(db_caption)
        db.commit()
        db.refresh(db_caption)
        return db_caption
    except Exception as e:
        db.rollback()
        raise e

def get_all_surveys(db: Session):
    """Get all surveys"""
    return db.query(models.Survey).all()

def update_survey_image_url(db: Session, survey_id: int, new_image_url: str):
    """Update survey image URL"""
    try:
        survey = db.query(models.Survey).filter(models.Survey.surveyId == survey_id).first()
        if survey:
            survey.imageUrl = new_image_url
            db.commit()
            db.refresh(survey)
            return survey
        return None
    except Exception as e:
        db.rollback()
        raise e

def create_agent_eval_detail_v2_samples(db: Session, caption_id: int, cultural_dist: dict, visual_dist: dict, hallucination_dist: dict, flag: int):
    """
    Create AgentEvalDetailV2 records from distribution data

    Args:
        db: Database session
        caption_id: Caption ID to associate with
        cultural_dist: Cultural distribution {'1': '0%', '2': '0%', '3': '20%', '4': '50%', '5': '30%'}
        visual_dist: Visual distribution
        hallucination_dist: Hallucination distribution
        flag: Flag value from API request

    Returns:
        List of created AgentEvalDetailV2 objects
    """
    print(f"🗄️ CRUD: Starting create_agent_eval_detail_v2_samples for caption {caption_id} with flag {flag}")
    print(f"🗄️ CRUD: Input distributions - cultural: {cultural_dist}, visual: {visual_dist}, hallucination: {hallucination_dist}")

    try:
        created_details = []

        # Process each metric type
        distributions = {
            'cultural': cultural_dist,
            'visual': visual_dist,
            'hallucination': hallucination_dist
        }

        print(f"🗄️ CRUD: Processing {len(distributions)} distribution types")

        for eval_type, distribution in distributions.items():
            print(f"🗄️ CRUD: Processing {eval_type} distribution: {distribution}")
            for likert_str, value_str in distribution.items():
                likert = int(likert_str)
                # Convert percentage string to float (remove % and convert)
                value = float(value_str.rstrip('%'))

                print(f"🗄️ CRUD: Creating v2 record - type: {eval_type}, likert: {likert}, value: {value}, flag: {flag}, captionId: {caption_id}")

                db_detail = models.AgentEvalDetailV2(
                    type=eval_type,
                    likert=likert,
                    value=value,
                    flag=flag,
                    captionId=caption_id
                )
                db.add(db_detail)
                created_details.append(db_detail)

        print(f"🗄️ CRUD: Created {len(created_details)} v2 records, committing to database...")
        db.commit()
        print(f"🗄️ CRUD: Database commit successful")

        # Refresh all created details
        for detail in created_details:
            db.refresh(detail)

        print(f"🗄️ CRUD: Successfully created and refreshed {len(created_details)} AgentEvalDetailV2 records")
        return created_details

    except Exception as e:
        print(f"🗄️ CRUD: Error occurred - {str(e)}")
        db.rollback()
        print(f"🗄️ CRUD: Database rolled back")
        raise e

def get_response_counts_by_survey(db: Session):
    """
    Get response count for each survey, ordered by count descending

    Returns:
        List of tuples: (Survey object, response_count)
    """
    from sqlalchemy import func

    results = (
        db.query(
            models.Survey,
            func.count(models.Response.responseId).label('response_count')
        )
        .join(models.Caption, models.Survey.surveyId == models.Caption.surveyId)
        .outerjoin(models.Response, models.Caption.captionId == models.Response.captionId)
        .group_by(models.Survey.surveyId)
        .order_by(func.count(models.Response.responseId).desc())
        .all()
    )

    return results

def get_captions_by_survey_titles(db: Session, survey_titles: list):
    """
    Get captions for surveys with specific titles

    Args:
        db: Database session
        survey_titles: List of survey titles to filter by

    Returns:
        List of Caption objects
    """
    return (
        db.query(models.Caption)
        .join(models.Survey, models.Caption.surveyId == models.Survey.surveyId)
        .filter(models.Survey.title.in_(survey_titles))
        .all()
    )

def delete_agent_eval_detail_unseen_by_caption_ids(db: Session, caption_ids: list):
    """
    Delete existing AgentEvalDetailUnseen records for given caption IDs

    Args:
        db: Database session
        caption_ids: List of caption IDs to delete evaluations for

    Returns:
        Number of deleted records
    """
    if not caption_ids:
        return 0

    deleted_count = (
        db.query(models.AgentEvalDetailUnseen)
        .filter(models.AgentEvalDetailUnseen.captionId.in_(caption_ids))
        .delete(synchronize_session=False)
    )
    db.commit()
    return deleted_count

def create_agent_eval_detail_unseen_samples(
    db: Session,
    caption_id: int,
    cultural_dist: dict,
    visual_dist: dict,
    hallucination_dist: dict,
    flag: int
):
    """
    Create AgentEvalDetailUnseen samples from evaluation distributions

    Args:
        db: Database session
        caption_id: Caption ID to associate with
        cultural_dist: Cultural distribution {'1': '0%', '2': '0%', '3': '20%', '4': '50%', '5': '30%'}
        visual_dist: Visual distribution
        hallucination_dist: Hallucination distribution
        flag: Flag value from API request

    Returns:
        List of created AgentEvalDetailUnseen objects
    """
    print(f"🗄️ CRUD: Starting create_agent_eval_detail_unseen_samples for caption {caption_id} with flag {flag}")
    print(f"🗄️ CRUD: Input distributions - cultural: {cultural_dist}, visual: {visual_dist}, hallucination: {hallucination_dist}")

    try:
        created_details = []

        # Process each metric type
        distributions = {
            'cultural': cultural_dist,
            'visual': visual_dist,
            'hallucination': hallucination_dist
        }

        print(f"🗄️ CRUD: Processing {len(distributions)} distribution types")

        for eval_type, distribution in distributions.items():
            print(f"🗄️ CRUD: Processing {eval_type} distribution: {distribution}")
            for likert_str, value_str in distribution.items():
                likert = int(likert_str)
                # Convert percentage string to float (remove % and convert)
                value = float(value_str.rstrip('%'))

                print(f"🗄️ CRUD: Creating unseen record - type: {eval_type}, likert: {likert}, value: {value}, flag: {flag}, captionId: {caption_id}")

                db_detail = models.AgentEvalDetailUnseen(
                    type=eval_type,
                    likert=likert,
                    value=value,
                    flag=flag,
                    captionId=caption_id
                )
                db.add(db_detail)
                created_details.append(db_detail)

        print(f"🗄️ CRUD: Created {len(created_details)} unseen records, committing to database...")
        db.commit()
        print(f"🗄️ CRUD: Database commit successful")

        # Refresh all created details
        for detail in created_details:
            db.refresh(detail)

        print(f"🗄️ CRUD: Successfully created and refreshed {len(created_details)} AgentEvalDetailUnseen records")
        return created_details

    except Exception as e:
        print(f"🗄️ CRUD: Error occurred - {str(e)}")
        db.rollback()
        print(f"🗄️ CRUD: Database rolled back")
        raise e

