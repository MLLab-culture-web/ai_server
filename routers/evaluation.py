
import base64
import json
import logging
import os
import numpy as np
import random
import glob
from typing import List, Tuple, Dict, Union
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
import requests
import ollama
from pydantic import BaseModel, validator
import re
import torch
from openai import OpenAI
from PIL import Image
import torchvision.transforms as transforms
import io
import urllib3

# SSL 경고 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 로깅 설정
def setup_evaluation_logger():
    """평가 로그를 위한 로거 설정"""
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)

    # 기존 핸들러 제거 (중복 방지)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 파일 핸들러 추가
    file_handler = logging.FileHandler('evaluate.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

# 전역 로거 인스턴스
eval_logger = setup_evaluation_logger()

# LPIPS import with fallback
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("LPIPS not available, using mock similarity calculation")
    LPIPS_AVAILABLE = False

import crud
import models
import schemas
from database import get_db

router = APIRouter()

MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5vl:72b")

# OpenAI client initialization
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

class ImageSimilarityCalculator:
    """이미지 유사도 계산기 (LPIPS 기반)"""

    def __init__(self):
        self.lpips_model = None
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        if LPIPS_AVAILABLE:
            try:
                self.lpips_model = lpips.LPIPS(net='alex', verbose=False)
                # GPU가 사용 가능하면 GPU로, 아니면 CPU로
                if torch.cuda.is_available():
                    self.lpips_model = self.lpips_model.cuda()
                else:
                    self.lpips_model = self.lpips_model.cpu()
                self.lpips_model.eval()  # 평가 모드로 설정
                print("LPIPS model loaded successfully")
            except Exception as e:
                print(f"Failed to load LPIPS model: {e}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                self.lpips_model = None

    def load_image_from_base64(self, base64_str: str) -> torch.Tensor:
        """Base64 문자열에서 이미지 텐서 로드"""
        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            tensor = self.transform(image).unsqueeze(0)

            # LPIPS 모델과 같은 device로 이동
            if self.lpips_model is not None:
                device = next(self.lpips_model.parameters()).device
                tensor = tensor.to(device)

            return tensor
        except Exception as e:
            print(f"Failed to load image from base64: {e}")
            raise e

    def calculate_image_similarity(self, base64_1: str, base64_2: str) -> float:
        """두 이미지의 LPIPS 유사도 계산"""
        if not self.lpips_model or not LPIPS_AVAILABLE:
            raise RuntimeError("LPIPS model is not available or not loaded properly")

        try:
            img1_tensor = self.load_image_from_base64(base64_1)
            img2_tensor = self.load_image_from_base64(base64_2)

            with torch.no_grad():
                lpips_distance = self.lpips_model(img1_tensor, img2_tensor)
                # 거리를 유사도로 변환 (LPIPS 거리는 0에 가까울수록 유사함)
                similarity = 1.0 - float(lpips_distance.item())
                # 유사도를 0~1 범위로 클램핑
                similarity = max(0.0, min(1.0, similarity))

            return similarity

        except Exception as e:
            print(f"LPIPS calculation failed: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            raise e

class EvaluationDistribution(BaseModel):
    """Pydantic model for validating Ollama evaluation response"""
    score_1: str = "0.0%"
    score_2: str = "0.0%"
    score_3: str = "0.0%"
    score_4: str = "0.0%"
    score_5: str = "0.0%"

    @validator('score_1', 'score_2', 'score_3', 'score_4', 'score_5')
    def validate_percentage(cls, v):
        if isinstance(v, (int, float)):
            return f"{float(v):.1f}%"
        if isinstance(v, str):
            # Handle various formats: "30.0%", "30%", "30.0", 30.0, etc.
            v = str(v).strip()
            if v.endswith('%'):
                v = v[:-1]
            try:
                float_val = float(v)
                return f"{float_val:.1f}%"
            except ValueError:
                return "0.0%"
        return "0.0%"

    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, int, float]]) -> 'EvaluationDistribution':
        """Create EvaluationDistribution from various dict formats"""
        return cls(
            score_1=data.get('1', data.get('score_1', "0.0%")),
            score_2=data.get('2', data.get('score_2', "0.0%")),
            score_3=data.get('3', data.get('score_3', "0.0%")),
            score_4=data.get('4', data.get('score_4', "0.0%")),
            score_5=data.get('5', data.get('score_5', "0.0%"))
        )

    def to_dict(self) -> Dict[str, str]:
        """Convert back to the expected dict format"""
        return {
            '1': self.score_1,
            '2': self.score_2,
            '3': self.score_3,
            '4': self.score_4,
            '5': self.score_5
        }

def calculate_weighted_score(distribution: Dict[str, str]) -> float:
    """
    Convert percentage distribution to weighted score (1-5 scale)

    Args:
        distribution: Dict with keys '1'-'5' and percentage values like '30.0%'

    Returns:
        Weighted average score between 1.0 and 5.0
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for score_str, percentage_str in distribution.items():
        try:
            score = int(score_str)
            # Remove % sign and convert to float
            percentage = float(percentage_str.rstrip('%'))

            weighted_sum += score * percentage
            total_weight += percentage
        except (ValueError, TypeError):
            continue

    if total_weight == 0:
        return 3.0  # Default middle score if no valid data

    return weighted_sum / total_weight


# Prompts from evaluation_agent_v2.py (simplified, no few-shot)

TASK_INSTRUCTION = """
I will provide a distribution over answer choices on a series of questions to help you understand the formatting of this task. After the examples, please express the distribution of answers for the question asked. Think of the distribution as a distribution of real people, and evaluate it as if it were real people. Don't be too sure or extreme in your judgments - most people give moderate scores between 2-4, with very few giving extreme scores (1 or 5).

Pay close attention to the human evaluation examples provided - they show you realistic patterns of how people actually evaluate similar content. Your distribution should reflect similar patterns and be grounded in common sense.

Do not follow the distribution expressed in the examples as they are only to give you an example of formatting, express the distribution based on your understanding of how humans may respond. Please only respond in the exact format of a dictionary mapping answer choice number to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted.

Consider a distribution that best represents realistic human evaluations with moderate, balanced judgments.
"""

CULTURAL_INPUT_PROMPT = """
Question: Evaluate how accurately and with sufficient detail this caption describes the cultural information of the artifact, such as its name, nation, and cultural context. 

Caption: {caption}
Artifact Info: {artifact_info}

Rating Scale:
'1'. Inaccurately Descriptive
'2'. Not Descriptive
'3'. Somewhat Descriptive
'4'. Mostly Descriptive
'5'. Highly Descriptive

Here are similar caption examples with actual human evaluation distributions:

Answer:
(We can get a distribution in JSON format)
"""

VISUAL_DETAIL_INPUT_PROMPT = """
Question: Evalute how accurately and specifically the caption describes the visuial elements of the artifact, such as shape, color, and othe observable features.

Caption: {caption}
Artifact Info: {artifact_info}

Rating Scale:

'1'. Inaccurately Descriptive
'2'. Not Descriptive
'3'. Somewhat Descriptive
'4'. Mostly Descriptive
'5'. highly Descriptive

Answer:
(We can get a distribution in JSON format)
"""

HALLUCINATION_INPUT_PROMPT = """
Question: Evaluate how much information that is not visually verifiable is included in this caption, such as knowledge and cultural conventions.

Caption: {caption}
Artifact Info: {artifact_info}

Rating Scale:
'1'. Not at all
'2'. Slithtly
'3'. Somewhat
'4'. Mostly
'5'. Completely

Answer:
"""

def _generate_embedding(text: str) -> List[float]:
    """OpenAI text-embedding-3-small 모델을 사용한 실제 임베딩 생성"""
    # try:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
    # except Exception as e:
    #     print(f"OpenAI embedding API call failed: {e}")
    #     print(f"Falling back to mock embedding for text: {text[:50]}...")
    #     # Fallback to mock embedding
    #     np.random.seed(hash(text) % (2**32))
    #     return np.random.normal(0, 1, 1536).tolist()  # text-embedding-3-small is 1536 dimensions

def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """코사인 유사도 계산"""
    a = np.array(embedding1)
    b = np.array(embedding2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_eval_metrics_from_json(json_filename: str) -> Dict:
    """JSON 파일에서 evalMetrics 데이터 로드"""
    try:
        if os.path.exists(json_filename):
            with open(json_filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {"total_count": 0, "metrics": []}
    except Exception as e:
        eval_logger.error(f"Failed to load JSON file {json_filename}: {e}")
        return {"total_count": 0, "metrics": []}

def save_eval_metrics_to_json(json_filename: str, metrics_data: Dict):
    """evalMetrics 데이터를 JSON 파일에 저장"""
    try:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        eval_logger.info(f"✅ Saved {metrics_data['total_count']} metrics to {json_filename}")
    except Exception as e:
        eval_logger.error(f"Failed to save JSON file {json_filename}: {e}")
        raise e

def find_latest_eval_metrics_file() -> str:
    """가장 최근의 evalMetrics JSON 파일을 찾기"""
    try:
        # evalMetrics_*.json 패턴으로 파일 검색
        pattern = "evalMetrics_*.json"
        metric_files = glob.glob(pattern)

        if not metric_files:
            eval_logger.warning("No evalMetrics JSON files found, using default filename")
            return "evalMetrics_20250922_125816.json"  # 기본 파일명

        # 파일명에서 타임스탬프 기준으로 정렬 (최신 파일이 마지막)
        metric_files.sort()
        latest_file = metric_files[-1]

        eval_logger.info(f"📂 Found {len(metric_files)} evalMetrics files, using latest: {latest_file}")
        return latest_file

    except Exception as e:
        eval_logger.error(f"Error finding latest evalMetrics file: {e}")
        return "evalMetrics_20250922_125816.json"  # 에러 시 기본 파일명 반환

def get_existing_pairs_from_json(metrics_data: Dict) -> set:
    """JSON 데이터에서 기존 쌍들을 추출"""
    existing_pairs = set()
    for metric in metrics_data.get("metrics", []):
        source_id = metric["sourceCaptionId"]
        target_id = metric["targetCaptionId"]
        existing_pairs.add((source_id, target_id))
    return existing_pairs

def remove_symmetric_duplicates_from_json(metrics_data: Dict) -> Dict:
    """JSON 데이터에서 대칭적인 중복 제거"""
    if not metrics_data.get("metrics"):
        return {"deleted_count": 0, "kept_count": 0, "total_unique_pairs": 0}

    metrics = metrics_data["metrics"]

    # Group metrics by symmetric pairs
    pair_groups = {}
    for metric in metrics:
        source_id = metric["sourceCaptionId"]
        target_id = metric["targetCaptionId"]

        # Create a canonical key (smaller ID first)
        if source_id < target_id:
            key = (source_id, target_id)
        else:
            key = (target_id, source_id)

        if key not in pair_groups:
            pair_groups[key] = []
        pair_groups[key].append(metric)

    # Find duplicates to remove and keep canonical forms
    new_metrics = []
    deleted_count = 0
    kept_count = 0

    for (canonical_source, canonical_target), group_metrics in pair_groups.items():
        if len(group_metrics) > 1:
            # Multiple records for the same symmetric pair
            keep_metric = None

            for metric in group_metrics:
                if metric["sourceCaptionId"] == canonical_source and metric["targetCaptionId"] == canonical_target:
                    # This is the canonical form (source < target)
                    if keep_metric is None:
                        keep_metric = metric
                    else:
                        deleted_count += 1
                else:
                    deleted_count += 1

            if keep_metric is None:
                # No canonical form found, convert first one to canonical
                keep_metric = group_metrics[0].copy()
                keep_metric["sourceCaptionId"] = canonical_source
                keep_metric["targetCaptionId"] = canonical_target
                deleted_count += len(group_metrics) - 1

            new_metrics.append(keep_metric)
            kept_count += 1
        else:
            # Single record, convert to canonical form if needed
            metric = group_metrics[0].copy()
            if metric["sourceCaptionId"] > metric["targetCaptionId"]:
                metric["sourceCaptionId"], metric["targetCaptionId"] = metric["targetCaptionId"], metric["sourceCaptionId"]
            new_metrics.append(metric)
            kept_count += 1

    # Update metrics data
    metrics_data["metrics"] = new_metrics
    metrics_data["total_count"] = len(new_metrics)

    return {
        "deleted_count": deleted_count,
        "kept_count": kept_count,
        "total_unique_pairs": len(pair_groups)
    }

def calculate_distribution_from_responses(responses: List[models.Response], metric: str) -> dict:
    """응답들로부터 분포 계산"""
    if not responses:
        return {"1": "0%", "2": "0%", "3": "0%", "4": "0%", "5": "0%"}

    counts = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}

    for response in responses:
        if metric == 'cultural':
            score = str(response.cultural)
        elif metric == 'visual':
            score = str(response.visual)
        else:  # hallucination
            score = str(response.hallucination)

        if score in counts:
            counts[score] += 1

    total = len(responses)
    distribution = {}
    for score in ["1", "2", "3", "4", "5"]:
        percentage = (counts[score] / total) * 100
        distribution[score] = f"{percentage:.1f}%"

    return distribution

def find_similar_captions_with_responses_json(db: Session, query_caption: str, query_image_base64: str, current_caption_id: int, current_survey_id: int, json_filename: str = "evalMetrics_20250922_125816.json", top_k: int = 5) -> List[Tuple[dict, float, dict]]:
    """JSON 파일에서 유사한 캡션들을 찾기"""

    eval_logger.info(f"🔍 Finding similar captions for ID {current_caption_id} (excluding same image survey ID: {current_survey_id}) using JSON metrics (bidirectional search)...")

    # JSON 파일에서 메트릭 로드
    metrics_data = load_eval_metrics_from_json(json_filename)
    if not metrics_data.get("metrics"):
        eval_logger.warning(f"No metrics found in JSON file {json_filename}. Please run /refresh-metric first.")
        return []

    similarities = []
    processed_caption_ids = set()  # 중복 방지를 위한 set

    # 현재 캡션 ID와 관련된 메트릭들 찾기 (양방향 검색)
    for metric in metrics_data["metrics"]:
        other_caption_id = None
        caption_similarity = 0.0
        image_similarity = 0.0

        # source -> target 방향 체크
        if metric["sourceCaptionId"] == current_caption_id:
            other_caption_id = metric["targetCaptionId"]
            caption_similarity = metric.get("captionSim", 0.0)
            image_similarity = metric.get("imageSim", 0.0)
        # target -> source 방향 체크 (역방향)
        elif metric["targetCaptionId"] == current_caption_id:
            other_caption_id = metric["sourceCaptionId"]
            caption_similarity = metric.get("captionSim", 0.0)
            image_similarity = metric.get("imageSim", 0.0)
        else:
            continue  # 현재 캡션과 관련 없음

        try:
            # 이미 처리된 캡션인지 확인 (중복 방지)
            if other_caption_id in processed_caption_ids:
                continue

            # 다른 캡션 정보 가져오기
            other_caption = (
                db.query(models.Caption)
                .options(joinedload(models.Caption.survey))
                .filter(models.Caption.captionId == other_caption_id)
                .first()
            )

            if not other_caption:
                continue

            # 같은 이미지 (survey) 제외
            if other_caption.surveyId == current_survey_id:
                continue

            # 응답이 있는 캡션인지 확인
            response_count = db.query(models.Response).filter(models.Response.captionId == other_caption.captionId).count()
            if response_count == 0:
                continue

            # 처리된 캡션 ID 추가
            processed_caption_ids.add(other_caption_id)

            # 종합 유사도: 캡션 유사도 × 이미지 유사도
            combined_similarity = caption_similarity * image_similarity

            eval_logger.info(f"Caption {other_caption.captionId}: caption_sim={caption_similarity:.3f}, image_sim={image_similarity:.3f}, combined={combined_similarity:.3f}")

            # 해당 캡션의 응답들 가져오기
            responses = db.query(models.Response).filter(models.Response.captionId == other_caption.captionId).all()

            # 각 지표별 분포 계산
            cultural_dist = calculate_distribution_from_responses(responses, 'cultural')
            visual_dist = calculate_distribution_from_responses(responses, 'visual')
            hallucination_dist = calculate_distribution_from_responses(responses, 'hallucination')

            caption_info = {
                "captionId": other_caption.captionId,
                "text": other_caption.text,
                "type": other_caption.type,
                "surveyId": other_caption.surveyId,
                "caption_similarity": caption_similarity,
                "image_similarity": image_similarity
            }

            response_distributions = {
                "cultural": cultural_dist,
                "visual": visual_dist,
                "hallucination": hallucination_dist
            }

            similarities.append((caption_info, combined_similarity, response_distributions))

        except Exception as e:
            eval_logger.error(f"Error processing JSON metric for captions {current_caption_id} <-> {other_caption_id}: {e}")
            continue

    # 종합 유사도 기준으로 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)

    eval_logger.info(f"📊 Top {min(top_k, len(similarities))} similar captions selected from JSON metrics (bidirectional)")
    for i, (caption_info, combined_score, _) in enumerate(similarities[:top_k]):
        eval_logger.info(f"  {i+1}. ID:{caption_info['captionId']} Score:{combined_score:.3f} (Cap:{caption_info.get('caption_similarity', 0):.3f} × Img:{caption_info.get('image_similarity', 0):.3f})")

    return similarities[:top_k]

def find_similar_captions_with_responses(db: Session, query_caption: str, query_image_base64: str, current_caption_id: int, current_survey_id: int, top_k: int = 5) -> List[Tuple[dict, float, dict]]:
    """유사한 캡션들을 찾기 - 최신 JSON 파일 자동 사용"""
    # 최신 JSON 파일 찾기
    latest_json_file = find_latest_eval_metrics_file()

    # 최신 JSON 파일 사용하여 유사도 검색
    return find_similar_captions_with_responses_json(
        db, query_caption, query_image_base64, current_caption_id, current_survey_id,
        latest_json_file, top_k
    )

def _download_image_as_base64(url_or_filename: str) -> str:
    """
    Download image from URL or load from local file system
    If url_or_filename is a filename (no http/https), load from local survey directory
    """
    try:
        # Check if it's a URL or local filename
        if url_or_filename.startswith(('http://', 'https://')):
            # Original URL download logic
            response = requests.get(url_or_filename, stream=True, timeout=30, verify=False)
            response.raise_for_status()
            image_data = response.content
        else:
            # Local file loading logic
            local_survey_dir = "/home/teom142/goinfre/culture/web/frontend/LMM/frontend/public/survey"
            file_path = os.path.join(local_survey_dir, url_or_filename)

            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file not found: {file_path}")

            # Read file content
            with open(file_path, 'rb') as f:
                image_data = f.read()

        return base64.b64encode(image_data).decode("utf-8")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {e}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Local image file not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")

def create_few_shot_examples(similar_captions_with_responses: List[Tuple[dict, float, dict]], metric_type: str) -> str:
    """유사한 캡션들의 실제 응답 분포로 few-shot examples 생성 (LPIPS 기반)"""
    examples = []

    # Don't add current caption as an example (자기 자신 제외)
    # Remove the current caption logic entirely

    if not similar_captions_with_responses:
        return "No similar examples found."

    # Add similar captions starting from example 1
    for i, (caption_info, _, distributions) in enumerate(similar_captions_with_responses, 1):
        distribution = distributions.get(metric_type, {"1": "0%", "2": "0%", "3": "0%", "4": "0%", "5": "0%"})

        example = f"""
# Example {i}
Caption: "{caption_info['text']}"
Human Distribution: {distribution}
"""
        examples.append(example)

    return "\n".join(examples)

def _parse_ollama_response(content: str) -> Dict[str, Union[str, int, float]]:
    """Parse Ollama response with multiple fallback strategies"""
    # Strategy 1: Try direct JSON parsing
    try:
        # Clean up common issues
        cleaned = content.strip()
        if cleaned.startswith('```python\n'):
            cleaned = cleaned[10:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]

        # Replace single quotes with double quotes for valid JSON
        cleaned = cleaned.replace("'", '"')
        return json.loads(cleaned)
    except:
        pass

    # Strategy 2: Extract dictionary using regex
    try:
        # Look for dictionary pattern {key: value, ...}
        dict_pattern = r'\{[^}]*\}'
        match = re.search(dict_pattern, content)
        if match:
            dict_str = match.group(0).replace("'", '"')
            return json.loads(dict_str)
    except:
        pass

    # Strategy 3: Extract percentages using regex
    try:
        result = {}
        # Look for patterns like '1': '20.0%' or "1": "20%"
        pattern = r"['\"]?(\d)['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)\%?['\"]?"
        matches = re.findall(pattern, content)
        for key, value in matches:
            result[key] = f"{float(value):.1f}%"

        if result and len(result) >= 3:  # Need at least some scores
            # Fill missing scores with 0%
            for i in range(1, 6):
                if str(i) not in result:
                    result[str(i)] = "0.0%"
            return result
    except:
        pass

    # Strategy 4: Look for just numbers that might represent percentages
    try:
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', content)
        if len(numbers) >= 5:
            result = {}
            for i, num in enumerate(numbers[:5], 1):
                result[str(i)] = f"{float(num):.1f}%"
            return result
    except:
        pass

    # Fallback: return default distribution
    return {"1": "0.0%", "2": "0.0%", "3": "20.0%", "4": "50.0%", "5": "30.0%"}

def _evaluate_metric(caption: str, image_base64: str, metric_type: str, few_shot_examples: str = "", artifact_info: dict = None) -> tuple[dict, dict]:
    # Extract artifact information
    artifact_context = ""
    if artifact_info:
        artifact_parts = []
        if artifact_info.get('title'):
            artifact_parts.append(f"Artifact Name: {artifact_info['title']}")
        if artifact_info.get('country'):
            artifact_parts.append(f"Country: {artifact_info['country']}")
        if artifact_info.get('category'):
            artifact_parts.append(f"Category: {artifact_info['category']}")

        if artifact_parts:
            artifact_context = f"\nArtifact Information:\n" + "\n".join(artifact_parts) + "\n"

    if metric_type == 'cultural':
        prompt = CULTURAL_INPUT_PROMPT.format(caption=caption, artifact_info=artifact_context)
    elif metric_type == 'visual':
        prompt = VISUAL_DETAIL_INPUT_PROMPT.format(caption=caption, artifact_info=artifact_context)
    else:  # hallucination
        prompt = HALLUCINATION_INPUT_PROMPT.format(caption=caption, artifact_info=artifact_context)

    content = ""
    debug_info = {
        "task": TASK_INSTRUCTION,
        "few_shot_examples": few_shot_examples,
        "prompt": prompt,
        "model": MODEL_NAME,
        "raw_response": "",
        "parsed_data": "",
        "validation_result": ""
    }

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    'role': 'system',
                    'content': TASK_INSTRUCTION
                },
                {
                    'role': 'system',
                    'content': few_shot_examples
                },
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_base64]
                }
            ]
        )
        content = response['message']['content']
        debug_info["raw_response"] = content

        # Parse the response using multiple strategies
        parsed_data = _parse_ollama_response(content)
        debug_info["parsed_data"] = parsed_data

        # Validate and normalize using Pydantic
        evaluation_dist = EvaluationDistribution.from_dict(parsed_data)
        debug_info["validation_result"] = "success"

        result_dict = evaluation_dist.to_dict()
        return result_dict, debug_info

    except Exception as e:
        eval_logger.error(f"Error evaluating metric {metric_type}: {e}")
        eval_logger.error(f"Ollama response content: {content}")
        debug_info["error"] = f"Parse error: {str(e)}"

        # Return fallback distribution
        fallback_dist = EvaluationDistribution()
        debug_info["validation_result"] = "fallback_used"
        return fallback_dist.to_dict(), debug_info


@router.post("/evaluate-all", response_model=schemas.EvaluateAllResponse)
def evaluate_all_captions(db: Session = Depends(get_db)):
    """
    Evaluate all captions and save results (without deleting existing data)
    """
    deleted_count = 0

    # 2. Get all captions with survey information
    all_captions = crud.get_all_captions_with_survey(db)

    total_captions = len(all_captions)
    evaluated_successfully = 0
    failed_evaluations = 0
    failed_caption_ids = []
    error_samples = []

    # 3. Evaluate each caption
    eval_logger.info(f"🚀 Starting evaluation of {total_captions} captions...")
    for idx, caption in enumerate(all_captions, 1):
        try:
            # Skip captions without survey or imageUrl
            if not caption.survey or not caption.survey.imageUrl:
                failed_evaluations += 1
                failed_caption_ids.append(caption.captionId)
                continue

            # Download and encode image
            try:
                image_base64 = _download_image_as_base64(caption.survey.imageUrl)
            except HTTPException as img_error:
                # 이미지 다운로드 실패 시 스킵하고 다음으로 진행
                eval_logger.error(f"Image download failed for caption {caption.captionId}: {img_error.detail}")
                failed_evaluations += 1
                failed_caption_ids.append(caption.captionId)

                if len(error_samples) < 5:
                    error_samples.append({
                        "caption_id": caption.captionId,
                        "error": img_error.detail,
                        "caption_text": caption.text[:100] + "..." if len(caption.text) > 100 else caption.text,
                        "survey_id": caption.surveyId,
                        "image_url": caption.survey.imageUrl,
                        "error_type": "ImageDownloadError"
                    })
                continue

            # 현재 캡션의 실제 인간 분포 가져오기
            current_responses = db.query(models.Response).filter(models.Response.captionId == caption.captionId).all()
            current_distributions = None
            if current_responses:
                current_distributions = {
                    'cultural': calculate_distribution_from_responses(current_responses, 'cultural'),
                    'visual': calculate_distribution_from_responses(current_responses, 'visual'),
                    'hallucination': calculate_distribution_from_responses(current_responses, 'hallucination')
                }
                eval_logger.info(f"📊 Found {len(current_responses)} human responses for current caption {caption.captionId}")
                eval_logger.info(f"📊 Current human distributions: {current_distributions}")
            else:
                eval_logger.info(f"📊 No human responses found for current caption {caption.captionId}")

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

            # Evaluate each metric with detailed logging
            eval_logger.info(f"🔍 === Evaluating Caption {caption.captionId} ({idx}/{total_captions}) ===")
            eval_logger.info(f"Caption text: {caption.text[:100]}{'...' if len(caption.text) > 100 else ''}")
            eval_logger.info(f"Survey ID: {caption.surveyId}")
            eval_logger.info(f"Image URL: {caption.survey.imageUrl}")

            # Log top 5 similar captions
            eval_logger.info(f"📊 Top 5 Similar Captions:")
            for i, (caption_info, combined_score, _) in enumerate(similar_captions_with_responses[:5], 1):
                eval_logger.info(f"  {i}. ID:{caption_info['captionId']} Score:{combined_score:.3f}")
                eval_logger.info(f"     Text: {caption_info['text'][:80]}{'...' if len(caption_info['text']) > 80 else ''}")
                eval_logger.info(f"     Cap_sim:{caption_info.get('caption_similarity', 0):.3f} × Img_sim:{caption_info.get('image_similarity', 0):.3f}")

            # Prepare artifact information
            artifact_info = {
                'title': caption.survey.title if caption.survey else None,
                'country': caption.survey.country if caption.survey else None,
                'category': caption.survey.category if caption.survey else None
            }

            cultural_result, cultural_debug = _evaluate_metric(caption.text, image_base64, 'cultural', cultural_examples, artifact_info)
            visual_result, visual_debug = _evaluate_metric(caption.text, image_base64, 'visual', visual_examples, artifact_info)
            hallucination_result, hallucination_debug = _evaluate_metric(caption.text, image_base64, 'hallucination', hallucination_examples, artifact_info)

            # Log evaluation results
            eval_logger.info(f"🎯 Evaluation Results:")
            eval_logger.info(f"Cultural distribution: {cultural_result}")
            eval_logger.info(f"Visual distribution: {visual_result}")
            eval_logger.info(f"Hallucination distribution: {hallucination_result}")

            # Log model responses (first 200 chars)
            eval_logger.info(f"🤖 Model Raw Responses:")
            eval_logger.info(f"Cultural: {cultural_debug.get('raw_response', 'N/A')[:200]}{'...' if len(cultural_debug.get('raw_response', '')) > 200 else ''}")
            eval_logger.info(f"Visual: {visual_debug.get('raw_response', 'N/A')[:200]}{'...' if len(visual_debug.get('raw_response', '')) > 200 else ''}")
            eval_logger.info(f"Hallucination: {hallucination_debug.get('raw_response', 'N/A')[:200]}{'...' if len(hallucination_debug.get('raw_response', '')) > 200 else ''}")

            # Log few-shot examples with detailed breakdown
            eval_logger.info(f"📝 Few-shot Examples Used:")

            def log_metric_examples(metric_name: str, examples: List[Tuple[dict, float, dict]], metric_type: str):
                eval_logger.info(f"  {metric_name} Examples ({len(examples)} examples):")
                for i, (caption_info, combined_similarity, distributions) in enumerate(examples, 1):
                    distribution = distributions.get(metric_type, {"1": "0%", "2": "0%", "3": "0%", "4": "0%", "5": "0%"})
                    caption_sim = caption_info.get('caption_similarity', 0)
                    image_sim = caption_info.get('image_similarity', 0)

                    eval_logger.info(f"    Example {i}:")
                    eval_logger.info(f"      Caption ID: {caption_info['captionId']}")
                    eval_logger.info(f"      Similarity Score: {combined_similarity:.3f} (Cap:{caption_sim:.3f} × Img:{image_sim:.3f})")
                    eval_logger.info(f"      Caption Text: \"{caption_info['text'][:120]}{'...' if len(caption_info['text']) > 120 else ''}\"")
                    eval_logger.info(f"      Human Distribution: {distribution}")

            # Log examples for each metric
            log_metric_examples("🏛️ Cultural", similar_captions_with_responses, 'cultural')
            log_metric_examples("👁️ Visual", similar_captions_with_responses, 'visual')
            log_metric_examples("🌟 Hallucination", similar_captions_with_responses, 'hallucination')

            evaluated_successfully += 1
            eval_logger.info(f"✅ Caption {caption.captionId} evaluated successfully! ({evaluated_successfully}/{total_captions} completed)")

        except Exception as e:
            error_info = {
                "caption_id": caption.captionId,
                "error": str(e),
                "caption_text": caption.text[:100] + "..." if len(caption.text) > 100 else caption.text,
                "survey_id": caption.surveyId,
                "image_url": caption.survey.imageUrl if caption.survey else None,
                "error_type": type(e).__name__
            }

            eval_logger.error(f"Failed to evaluate caption {caption.captionId}: {e}")
            eval_logger.error(f"Caption text: {caption.text[:100]}...")
            eval_logger.error(f"Survey ID: {caption.surveyId}")
            eval_logger.error(f"Image URL: {caption.survey.imageUrl if caption.survey else 'None'}")
            import traceback
            eval_logger.error(f"Full traceback: {traceback.format_exc()}")

            failed_evaluations += 1
            failed_caption_ids.append(caption.captionId)

            # 처음 5개 에러만 샘플로 저장
            if len(error_samples) < 5:
                error_samples.append(error_info)

    # 4. Prepare summary
    summary = {
        "deleted_previous_agent_evaluations": deleted_count,
        "processing_results": {
            "total": total_captions,
            "successful": evaluated_successfully,
            "failed": failed_evaluations
        }
    }

    if failed_caption_ids:
        summary["failed_caption_ids"] = failed_caption_ids[:10]  # Show first 10 failed IDs

    # Log final summary
    eval_logger.info(f"🏁 === Evaluation Complete ===")
    eval_logger.info(f"📊 Total captions processed: {total_captions}")
    eval_logger.info(f"✅ Successfully evaluated: {evaluated_successfully}")
    eval_logger.info(f"❌ Failed evaluations: {failed_evaluations}")
    eval_logger.info(f"🗑️ Deleted previous evaluations: {deleted_count}")
    if failed_caption_ids:
        eval_logger.info(f"🔍 Failed caption IDs (first 10): {failed_caption_ids[:10]}")
    if error_samples:
        eval_logger.info(f"⚠️ Error samples: {len(error_samples)} errors logged")

    return schemas.EvaluateAllResponse(
        total_captions=total_captions,
        evaluated_successfully=evaluated_successfully,
        failed_evaluations=failed_evaluations,
        deleted_previous_evaluations=deleted_count,
        summary=summary,
        error_samples=error_samples
    )

@router.post("/evaluate", response_model=schemas.EvaluationResponse)
def evaluate_caption_range(
    request: schemas.EvaluationRequest,
    db: Session = Depends(get_db)
):
    start_caption_id = request.startCaptionId
    end_caption_id = request.endCaptionId

    # 입력 검증
    if start_caption_id > end_caption_id:
        raise HTTPException(status_code=400, detail="startCaptionId must be less than or equal to endCaptionId")

    eval_logger.info(f"🚀 Starting evaluation for caption range {start_caption_id} to {end_caption_id}")

    # 범위 내 캡션들 가져오기
    captions_in_range = (
        db.query(models.Caption)
        .options(joinedload(models.Caption.survey))
        .filter(models.Caption.captionId >= start_caption_id)
        .filter(models.Caption.captionId <= end_caption_id)
        .all()
    )

    if not captions_in_range:
        raise HTTPException(status_code=404, detail=f"No captions found in range {start_caption_id} to {end_caption_id}")

    total_captions = len(captions_in_range)
    evaluated_successfully = 0
    failed_evaluations = 0
    results = []
    error_samples = []

    eval_logger.info(f"📊 Found {total_captions} captions in range")

    # 각 캡션에 대해 평가 수행
    for idx, caption in enumerate(captions_in_range, 1):
        try:
            eval_logger.info(f"🔍 Evaluating caption {caption.captionId} ({idx}/{total_captions})")

            # 캡션과 관련된 survey/image 검증
            if not caption.survey or not caption.survey.imageUrl:
                eval_logger.error(f"Image URL not found for caption {caption.captionId}")
                failed_evaluations += 1
                if len(error_samples) < 5:
                    error_samples.append({
                        "caption_id": caption.captionId,
                        "error": "Image URL not found for the caption's survey",
                        "error_type": "MissingImageURL"
                    })
                continue

            image_url = caption.survey.imageUrl
            caption_text = caption.text

            # 이미지 다운로드 및 인코딩
            try:
                image_base64 = _download_image_as_base64(image_url)
            except HTTPException as img_error:
                eval_logger.error(f"Image download failed for caption {caption.captionId}: {img_error.detail}")
                failed_evaluations += 1
                if len(error_samples) < 5:
                    error_samples.append({
                        "caption_id": caption.captionId,
                        "error": img_error.detail,
                        "error_type": "ImageDownloadError"
                    })
                continue

            # 현재 캡션의 실제 인간 분포 가져오기
            current_responses = db.query(models.Response).filter(models.Response.captionId == caption.captionId).all()
            current_distributions = None
            if current_responses:
                current_distributions = {
                    'cultural': calculate_distribution_from_responses(current_responses, 'cultural'),
                    'visual': calculate_distribution_from_responses(current_responses, 'visual'),
                    'hallucination': calculate_distribution_from_responses(current_responses, 'hallucination')
                }
                eval_logger.info(f"📊 Found {len(current_responses)} human responses for current caption {caption.captionId}")
                eval_logger.info(f"📊 Current human distributions: {current_distributions}")
            else:
                eval_logger.info(f"📊 No human responses found for current caption {caption.captionId}")

            # 유사한 캡션들 찾기 - 최신 JSON 파일 사용
            latest_json_file = find_latest_eval_metrics_file()
            similar_captions_with_responses = find_similar_captions_with_responses_json(
                db, caption_text, image_base64, caption.captionId, caption.surveyId,
                latest_json_file, top_k=5
            )

            # 각 메트릭 평가 (자기 자신 제외)
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

            cultural_result, cultural_debug = _evaluate_metric(caption_text, image_base64, 'cultural', cultural_examples, artifact_info)
            visual_result, visual_debug = _evaluate_metric(caption_text, image_base64, 'visual', visual_examples, artifact_info)
            hallucination_result, hallucination_debug = _evaluate_metric(caption_text, image_base64, 'hallucination', hallucination_examples, artifact_info)

            # 결과 저장
            caption_result = {
                "caption_id": caption.captionId,
                "caption_text": caption_text,
                "image_url": image_url,
                "cultural": cultural_result,
                "visual": visual_result,
                "hallucination": hallucination_result,
                "similar_captions_count": len(similar_captions_with_responses),
                "debug": {
                    "cultural": cultural_debug,
                    "visual": visual_debug,
                    "hallucination": hallucination_debug
                }
            }
            results.append(caption_result)
            evaluated_successfully += 1

            eval_logger.info(f"✅ Successfully evaluated caption {caption.captionId}")

        except Exception as e:
            eval_logger.error(f"Failed to evaluate caption {caption.captionId}: {e}")
            failed_evaluations += 1

            if len(error_samples) < 5:
                error_samples.append({
                    "caption_id": caption.captionId,
                    "error": str(e),
                    "error_type": type(e).__name__
                })

    # 요약 정보
    summary = {
        "start_caption_id": start_caption_id,
        "end_caption_id": end_caption_id,
        "captions_found": total_captions,
        "evaluated_successfully": evaluated_successfully,
        "failed_evaluations": failed_evaluations
    }

    eval_logger.info(f"🏁 Evaluation complete: {evaluated_successfully}/{total_captions} captions successful")

    # 결과 반환
    return schemas.EvaluationResponse(
        start_caption_id=start_caption_id,
        end_caption_id=end_caption_id,
        total_captions=total_captions,
        evaluated_successfully=evaluated_successfully,
        failed_evaluations=failed_evaluations,
        results=results,
        summary=summary,
        error_samples=error_samples
    )

@router.post("/refresh-metric", response_model=schemas.RefreshMetricResponse)
def refresh_similarity_metrics(db: Session = Depends(get_db)):
    """
    Calculate and store all caption-to-caption similarity metrics (incremental) - JSON based
    """
    eval_logger.info("🚀 Starting incremental similarity metrics calculation (JSON mode)...")

    # 0. Generate JSON filename and load existing data
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"evalMetrics_{timestamp}.json"

    # Load existing JSON file or create new structure
    existing_json_filename = find_latest_eval_metrics_file()
    eval_logger.info(f"📂 Loading existing data from {existing_json_filename}...")

    current_metrics = load_eval_metrics_from_json(existing_json_filename)
    eval_logger.info(f"📊 Loaded {current_metrics['total_count']} existing metrics")

    # 1. Remove symmetric duplicates from JSON data
    eval_logger.info("🧹 Removing symmetric duplicate pairs from JSON data...")
    try:
        duplicate_removal_result = remove_symmetric_duplicates_from_json(current_metrics)
        eval_logger.info(f"🗑️ Duplicate removal complete:")
        eval_logger.info(f"   - Deleted duplicates: {duplicate_removal_result['deleted_count']}")
        eval_logger.info(f"   - Kept unique pairs: {duplicate_removal_result['kept_count']}")
        eval_logger.info(f"   - Total unique pairs: {duplicate_removal_result['total_unique_pairs']}")
    except Exception as cleanup_error:
        eval_logger.error(f"Failed to remove duplicates: {cleanup_error}")
        # Continue with the process even if duplicate removal fails

    # 2. Get existing pairs from JSON for fast lookup
    existing_pairs = get_existing_pairs_from_json(current_metrics)
    existing_count = len(existing_pairs)
    eval_logger.info(f"📊 Found {existing_count} existing metric pairs in JSON")

    # 3. Get all captions with survey information
    all_captions = crud.get_all_captions_with_survey(db)

    # Filter captions with valid surveys and imageUrls
    valid_captions = [
        caption for caption in all_captions
        if caption.survey and caption.survey.imageUrl
    ]

    total_captions = len(valid_captions)
    total_pairs = total_captions * (total_captions - 1)  # N*(N-1) pairs (excluding self)
    pairs_to_process = total_pairs - existing_count

    eval_logger.info(f"📊 Processing {total_captions} captions")
    eval_logger.info(f"📊 Total possible pairs: {total_pairs}")
    eval_logger.info(f"📊 Already calculated: {existing_count}")
    eval_logger.info(f"📊 Pairs to process: {pairs_to_process}")

    processed_successfully = 0
    skipped_existing = 0
    failed_calculations = 0
    error_samples = []

    # 이미지 유사도 계산기 초기화
    image_sim_calc = ImageSimilarityCalculator()

    # New metrics list for JSON storage
    new_metrics = []

    for i, source_caption in enumerate(valid_captions, 1):
        try:
            eval_logger.info(f"🔍 Processing caption {source_caption.captionId} ({i}/{total_captions})")
            caption_processed_count = 0
            caption_skipped_count = 0
            caption_failed_count = 0

            # Download source image and generate embedding once per source caption
            try:
                source_image_base64 = _download_image_as_base64(source_caption.survey.imageUrl)
                source_embedding = _generate_embedding(source_caption.text)
            except Exception as source_error:
                eval_logger.error(f"Failed to process source caption {source_caption.captionId}: {source_error}")
                failed_calculations += len(valid_captions) - 1  # Failed for all pairs with this caption
                continue

            for target_caption in valid_captions:
                # Skip self-comparison
                if source_caption.captionId == target_caption.captionId:
                    continue

                # Skip if already exists (check both directions since similarity is symmetric)
                pair_key = (source_caption.captionId, target_caption.captionId)
                reverse_pair_key = (target_caption.captionId, source_caption.captionId)
                if pair_key in existing_pairs or reverse_pair_key in existing_pairs:
                    eval_logger.info(f"  ⏭️ Pair {source_caption.captionId} -> {target_caption.captionId}: skipped (already exists)")
                    skipped_existing += 1
                    caption_skipped_count += 1
                    continue

                try:
                    # Calculate caption similarity
                    target_embedding = _generate_embedding(target_caption.text)
                    caption_similarity = cosine_similarity(source_embedding, target_embedding)

                    # Calculate image similarity
                    target_image_base64 = _download_image_as_base64(target_caption.survey.imageUrl)
                    image_similarity = image_sim_calc.calculate_image_similarity(
                        source_image_base64, target_image_base64
                    )

                    # Log individual pair processing
                    eval_logger.info(f"  ✅ Pair {source_caption.captionId} -> {target_caption.captionId}: cap_sim={caption_similarity:.3f}, img_sim={image_similarity:.3f}")

                    # Generate unique ID based on current total count
                    new_id = len(current_metrics["metrics"]) + 1

                    # Add to new metrics list for JSON storage
                    new_metric = {
                        "id": new_id,
                        "sourceCaptionId": source_caption.captionId,
                        "targetCaptionId": target_caption.captionId,
                        "captionSim": float(caption_similarity),
                        "imageSim": float(image_similarity),
                        "createdAt": None
                    }
                    new_metrics.append(new_metric)

                    # Add metric to current_metrics and save immediately
                    current_metrics["metrics"].append(new_metric)
                    current_metrics["total_count"] = len(current_metrics["metrics"])

                    # Save to JSON file immediately
                    try:
                        save_eval_metrics_to_json(json_filename, current_metrics)
                        eval_logger.info(f"💾 Saved metric {source_caption.captionId} -> {target_caption.captionId} to JSON")
                    except Exception as save_error:
                        eval_logger.error(f"Failed to save metric to JSON: {save_error}")

                    # Add both directions to existing_pairs to avoid duplicates in this session
                    existing_pairs.add(pair_key)
                    existing_pairs.add(reverse_pair_key)

                    processed_successfully += 1
                    caption_processed_count += 1

                except Exception as pair_error:
                    eval_logger.error(f"  ❌ Pair {source_caption.captionId} -> {target_caption.captionId}: failed - {pair_error}")
                    failed_calculations += 1
                    caption_failed_count += 1

                    if len(error_samples) < 5:
                        error_samples.append({
                            "source_caption_id": source_caption.captionId,
                            "target_caption_id": target_caption.captionId,
                            "error": str(pair_error),
                            "error_type": type(pair_error).__name__
                        })

            # Log completion for this caption
            eval_logger.info(f"✅ Completed caption {source_caption.captionId}: processed={caption_processed_count}, skipped={caption_skipped_count}, failed={caption_failed_count}")

        except Exception as caption_error:
            eval_logger.error(f"Failed to process caption {source_caption.captionId}: {caption_error}")
            # Count only the pairs that would have been processed (not already existing)
            remaining_targets = len([t for t in valid_captions
                                  if t.captionId != source_caption.captionId
                                  and (source_caption.captionId, t.captionId) not in existing_pairs
                                  and (t.captionId, source_caption.captionId) not in existing_pairs])
            failed_calculations += remaining_targets

    # Final save summary
    if new_metrics:
        eval_logger.info(f"✅ Total {len(new_metrics)} new metrics saved incrementally to JSON file")
    else:
        # Save existing data to new JSON file (for backup)
        save_eval_metrics_to_json(json_filename, current_metrics)
        eval_logger.info(f"📄 No new metrics added, saved existing data for backup")

    # Summary
    summary = {
        "total_captions": total_captions,
        "total_pairs": total_pairs,
        "existing_pairs": existing_count,
        "pairs_to_process": pairs_to_process,
        "processed_successfully": processed_successfully,
        "skipped_existing": skipped_existing,
        "failed_calculations": failed_calculations
    }

    eval_logger.info(f"🏁 === Incremental Similarity Metrics Calculation Complete ===")
    eval_logger.info(f"📊 Total caption pairs: {total_pairs}")
    eval_logger.info(f"🔄 Skipped existing pairs: {skipped_existing}")
    eval_logger.info(f"✅ Successfully processed new pairs: {processed_successfully}")
    eval_logger.info(f"❌ Failed calculations: {failed_calculations}")

    # Add JSON file info to summary
    summary["eval_metrics_json_file"] = json_filename

    return schemas.RefreshMetricResponse(
        total_caption_pairs=total_pairs,
        processed_successfully=processed_successfully,
        failed_calculations=failed_calculations,
        deleted_previous_metrics=0,  # No deletion in incremental mode
        summary=summary,
        error_samples=error_samples
    )

@router.post("/evaluate-with-flag-all", response_model=schemas.EvaluateAllResponse)
def evaluate_all_captions_with_flags(
    request: schemas.EvaluateAllWithFlagRequest = schemas.EvaluateAllWithFlagRequest(),
    db: Session = Depends(get_db)
):
    """
    Evaluate all captions and save to both AgentEvalDetail and AgentEvalDetail_v2 tables
    Flag value increases by 50 for every 50 responses based on response ID
    Can specify start_flag to resume from a specific flag value
    """
    start_flag = request.start_flag
    eval_logger.info(f"🚀 Starting evaluate-with-flag-all: evaluating all captions with incremental flags (starting from flag {start_flag})")

    # Get all captions with survey information
    all_captions = crud.get_all_captions_with_survey(db)
    total_captions = len(all_captions)
    evaluated_successfully = 0
    failed_evaluations = 0
    failed_caption_ids = []
    error_samples = []

    # Get all responses ordered by ID to calculate actual data-based thresholds
    all_responses = db.query(models.Response).order_by(models.Response.responseId.asc()).all()
    total_response_count = len(all_responses)
    eval_logger.info(f"📊 Total responses in database: {total_response_count}")

    # Calculate how many flag values we need to generate
    # Flag starts at 50 and increases by 50 for every 50 actual responses
    flag_values = []
    current_flag = 50
    response_index = 50  # First flag at 50 responses

    while response_index <= total_response_count:
        # Get the response ID at this index (50th, 100th, 150th... response)
        response_threshold_id = all_responses[response_index - 1].responseId
        flag_values.append((response_threshold_id, current_flag))
        response_index += 50
        current_flag += 50

    # Filter flag_values to only include flags >= start_flag
    flag_values = [(threshold, flag) for threshold, flag in flag_values if flag >= start_flag]

    eval_logger.info(f"📊 Flag values to generate: {flag_values}")
    eval_logger.info(f"📊 Total flag configurations: {len(flag_values)}")

    if not flag_values:
        eval_logger.warning(f"⚠️ No flag values to generate (no flags >= {start_flag})")
        return schemas.EvaluateAllResponse(
            total_captions=total_captions,
            evaluated_successfully=0,
            failed_evaluations=0,
            deleted_previous_evaluations=0,
            summary={"message": f"No flag values >= {start_flag} to process"},
            error_samples=[]
        )

    eval_logger.info(f"🚀 Starting evaluation of {total_captions} captions with {len(flag_values)} flag configurations (from flag {start_flag})...")

    for flag_threshold, flag_value in flag_values:
        eval_logger.info(f"🏷️ === Processing Flag {flag_value} (Response threshold: {flag_threshold}) ===")

        # Process each caption for this flag
        for idx, caption in enumerate(all_captions, 1):
            try:
                # Skip captions without survey or imageUrl
                if not caption.survey or not caption.survey.imageUrl:
                    eval_logger.error(f"Image URL not found for caption {caption.captionId}")
                    failed_evaluations += 1
                    failed_caption_ids.append(caption.captionId)
                    continue

                # Download and encode image
                try:
                    image_base64 = _download_image_as_base64(caption.survey.imageUrl)
                except HTTPException as img_error:
                    eval_logger.error(f"Image download failed for caption {caption.captionId}: {img_error.detail}")
                    failed_evaluations += 1
                    failed_caption_ids.append(caption.captionId)
                    continue

                # Get current caption's human distributions using responses up to the flag threshold
                current_responses = (
                    db.query(models.Response)
                    .filter(models.Response.captionId == caption.captionId)
                    .filter(models.Response.responseId <= flag_threshold)
                    .all()
                )

                current_distributions = None
                if current_responses:
                    current_distributions = {
                        'cultural': calculate_distribution_from_responses(current_responses, 'cultural'),
                        'visual': calculate_distribution_from_responses(current_responses, 'visual'),
                        'hallucination': calculate_distribution_from_responses(current_responses, 'hallucination')
                    }
                    eval_logger.info(f"📊 Found {len(current_responses)} human responses for caption {caption.captionId} up to response ID {flag_threshold}")

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

                # Save to both AgentEvalDetail and AgentEvalDetail_v2 tables
                try:
                    # Save to original AgentEvalDetail table
                    created_details_v1 = crud.create_agent_eval_detail_samples(
                        db=db,
                        caption_id=caption.captionId,
                        cultural_dist=cultural_result,
                        visual_dist=visual_result,
                        hallucination_dist=hallucination_result,
                        flag=flag_value
                    )
                    eval_logger.info(f"✅ Saved {len(created_details_v1)} evaluation details to AgentEvalDetail for caption {caption.captionId} with flag {flag_value}")

                    # Save to AgentEvalDetail_v2 table
                    created_details_v2 = crud.create_agent_eval_detail_v2_samples(
                        db=db,
                        caption_id=caption.captionId,
                        cultural_dist=cultural_result,
                        visual_dist=visual_result,
                        hallucination_dist=hallucination_result,
                        flag=flag_value
                    )
                    eval_logger.info(f"✅ Saved {len(created_details_v2)} evaluation details to AgentEvalDetail_v2 for caption {caption.captionId} with flag {flag_value}")

                except Exception as db_error:
                    eval_logger.error(f"Failed to save agent evaluation details for caption {caption.captionId} with flag {flag_value}: {db_error}")
                    failed_evaluations += 1
                    continue

                eval_logger.info(f"✅ Caption {caption.captionId} evaluated successfully with flag {flag_value}! ({idx}/{total_captions} captions for this flag)")

            except Exception as e:
                eval_logger.error(f"Failed to evaluate caption {caption.captionId} with flag {flag_value}: {e}")
                failed_evaluations += 1
                if caption.captionId not in failed_caption_ids:
                    failed_caption_ids.append(caption.captionId)

                if len(error_samples) < 5:
                    error_samples.append({
                        "caption_id": caption.captionId,
                        "flag": flag_value,
                        "error": str(e),
                        "error_type": type(e).__name__
                    })

        eval_logger.info(f"✅ Completed processing flag {flag_value} for all {total_captions} captions")
        evaluated_successfully += total_captions - len([c for c in all_captions if c.captionId in failed_caption_ids])

    # Prepare summary
    total_operations = total_captions * len(flag_values)
    summary = {
        "total_captions": total_captions,
        "flag_configurations": len(flag_values),
        "start_flag": start_flag,
        "flag_values_generated": [flag for _, flag in flag_values],
        "total_operations": total_operations,
        "total_response_count": total_response_count,
        "response_thresholds": [threshold for threshold, _ in flag_values]
    }

    eval_logger.info(f"🏁 === Evaluate-with-flag-all Complete ===")
    eval_logger.info(f"📊 Total captions: {total_captions}")
    eval_logger.info(f"📊 Flag configurations: {len(flag_values)}")
    eval_logger.info(f"📊 Total operations: {total_operations}")
    eval_logger.info(f"✅ Successfully completed: {evaluated_successfully}")
    eval_logger.info(f"❌ Failed operations: {failed_evaluations}")

    return schemas.EvaluateAllResponse(
        total_captions=total_captions,
        evaluated_successfully=evaluated_successfully,
        failed_evaluations=failed_evaluations,
        deleted_previous_evaluations=0,
        summary=summary,
        error_samples=error_samples
    )

@router.post("/evaluate-with-flag", response_model=schemas.EvaluationResponse)
def evaluate_caption_range_with_flag(
    request: schemas.EvaluationRequestWithFlag,
    db: Session = Depends(get_db)
):
    """
    Evaluate captions and save to AgentEvalDetail table with flag
    This uses the new data structure where:
    - type: 'cultural', 'visual', 'hallucination'
    - likert: 1-5 (from distribution keys)
    - value: percentage value (from distribution values)
    - flag: from API request
    """
    start_caption_id = request.startCaptionId
    end_caption_id = request.endCaptionId
    flag = request.flag

    # 입력 검증
    if start_caption_id > end_caption_id:
        raise HTTPException(status_code=400, detail="startCaptionId must be less than or equal to endCaptionId")

    eval_logger.info(f"🚀 Starting evaluation with flag {flag} for caption range {start_caption_id} to {end_caption_id}")

    # 범위 내 캡션들 가져오기
    captions_in_range = (
        db.query(models.Caption)
        .options(joinedload(models.Caption.survey))
        .filter(models.Caption.captionId >= start_caption_id)
        .filter(models.Caption.captionId <= end_caption_id)
        .all()
    )

    if not captions_in_range:
        raise HTTPException(status_code=404, detail=f"No captions found in range {start_caption_id} to {end_caption_id}")

    total_captions = len(captions_in_range)
    evaluated_successfully = 0
    failed_evaluations = 0
    results = []
    error_samples = []

    eval_logger.info(f"📊 Found {total_captions} captions in range")

    # 각 캡션에 대해 평가 수행
    for idx, caption in enumerate(captions_in_range, 1):
        try:
            eval_logger.info(f"🔍 Evaluating caption {caption.captionId} ({idx}/{total_captions}) with flag {flag}")

            # 캡션과 관련된 survey/image 검증
            if not caption.survey or not caption.survey.imageUrl:
                eval_logger.error(f"Image URL not found for caption {caption.captionId}")
                failed_evaluations += 1
                if len(error_samples) < 5:
                    error_samples.append({
                        "caption_id": caption.captionId,
                        "error": "Image URL not found for the caption's survey",
                        "error_type": "MissingImageURL"
                    })
                continue

            image_url = caption.survey.imageUrl
            caption_text = caption.text

            # 이미지 다운로드 및 인코딩
            try:
                image_base64 = _download_image_as_base64(image_url)
            except HTTPException as img_error:
                eval_logger.error(f"Image download failed for caption {caption.captionId}: {img_error.detail}")
                failed_evaluations += 1
                if len(error_samples) < 5:
                    error_samples.append({
                        "caption_id": caption.captionId,
                        "error": img_error.detail,
                        "error_type": "ImageDownloadError"
                    })
                continue

            # 현재 캡션의 실제 인간 분포 가져오기
            current_responses = db.query(models.Response).filter(models.Response.captionId == caption.captionId).all()
            current_distributions = None
            if current_responses:
                current_distributions = {
                    'cultural': calculate_distribution_from_responses(current_responses, 'cultural'),
                    'visual': calculate_distribution_from_responses(current_responses, 'visual'),
                    'hallucination': calculate_distribution_from_responses(current_responses, 'hallucination')
                }
                eval_logger.info(f"📊 Found {len(current_responses)} human responses for current caption {caption.captionId}")
                eval_logger.info(f"📊 Current human distributions: {current_distributions}")
            else:
                eval_logger.info(f"📊 No human responses found for current caption {caption.captionId}")

            # 유사한 캡션들 찾기 - 최신 JSON 파일 사용
            latest_json_file = find_latest_eval_metrics_file()
            similar_captions_with_responses = find_similar_captions_with_responses_json(
                db, caption_text, image_base64, caption.captionId, caption.surveyId,
                latest_json_file, top_k=5
            )

            # 각 메트릭 평가 (자기 자신 제외)
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

            cultural_result, cultural_debug = _evaluate_metric(caption_text, image_base64, 'cultural', cultural_examples, artifact_info)
            visual_result, visual_debug = _evaluate_metric(caption_text, image_base64, 'visual', visual_examples, artifact_info)
            hallucination_result, hallucination_debug = _evaluate_metric(caption_text, image_base64, 'hallucination', hallucination_examples, artifact_info)

            # 새로운 AgentEvalDetail 테이블에 저장
            database_save_status = "success"
            try:
                created_details = crud.create_agent_eval_detail_samples(
                    db=db,
                    caption_id=caption.captionId,
                    cultural_dist=cultural_result,
                    visual_dist=visual_result,
                    hallucination_dist=hallucination_result,
                    flag=flag
                )
                eval_logger.info(f"✅ Saved {len(created_details)} evaluation details for caption {caption.captionId} with flag {flag}")
            except Exception as db_error:
                eval_logger.error(f"Failed to save agent evaluation details for caption {caption.captionId}: {db_error}")
                database_save_status = f"failed: {str(db_error)}"

            # 결과 저장
            caption_result = {
                "caption_id": caption.captionId,
                "caption_text": caption_text,
                "image_url": image_url,
                "cultural": cultural_result,
                "visual": visual_result,
                "hallucination": hallucination_result,
                "flag": flag,
                "database_save": database_save_status,
                "similar_captions_count": len(similar_captions_with_responses),
                "debug": {
                    "cultural": cultural_debug,
                    "visual": visual_debug,
                    "hallucination": hallucination_debug
                }
            }
            results.append(caption_result)
            evaluated_successfully += 1

            eval_logger.info(f"✅ Successfully evaluated caption {caption.captionId} with flag {flag}")

        except Exception as e:
            eval_logger.error(f"Failed to evaluate caption {caption.captionId}: {e}")
            failed_evaluations += 1

            if len(error_samples) < 5:
                error_samples.append({
                    "caption_id": caption.captionId,
                    "error": str(e),
                    "error_type": type(e).__name__
                })

    # 요약 정보
    summary = {
        "start_caption_id": start_caption_id,
        "end_caption_id": end_caption_id,
        "flag": flag,
        "captions_found": total_captions,
        "evaluated_successfully": evaluated_successfully,
        "failed_evaluations": failed_evaluations,
        "data_structure": "AgentEvalDetail"
    }

    eval_logger.info(f"🏁 Evaluation with flag {flag} complete: {evaluated_successfully}/{total_captions} captions successful")

    # 결과 반환
    return schemas.EvaluationResponse(
        start_caption_id=start_caption_id,
        end_caption_id=end_caption_id,
        total_captions=total_captions,
        evaluated_successfully=evaluated_successfully,
        failed_evaluations=failed_evaluations,
        results=results,
        summary=summary,
        error_samples=error_samples
    )

@router.get("/variance-analysis-all", response_model=schemas.VarianceAnalysisResponse)
def variance_analysis_all(db: Session = Depends(get_db)):
    """
    분산 분석 API:
    - wasserstein: 카테고리별 평균, 분산
    - user response: 평균, 분산 (전체 카테고리 통틀어서, user 수는 20으로 고정)
    - 카테고리별 응답 분포 (카운트)
    """
    from scipy.stats import wasserstein_distance
    from collections import defaultdict

    # 모든 응답과 관련 survey/caption 정보 가져오기
    responses = (
        db.query(models.Response)
        .options(
            joinedload(models.Response.caption).joinedload(models.Caption.survey)
        )
        .all()
    )

    # 카테고리별로 wasserstein distance 계산을 위한 데이터 수집
    category_wasserstein_data = defaultdict(list)
    category_response_counts = defaultdict(int)
    user_response_counts = defaultdict(int)

    for response in responses:
        if response.caption and response.caption.survey:
            category = response.caption.survey.category

            # Wasserstein distance 계산을 위해 각 응답의 likert 점수들 저장
            category_wasserstein_data[category].extend([
                response.cultural,
                response.visual,
                response.hallucination
            ])

            # 카테고리별 응답 수 카운트
            category_response_counts[category] += 1

            # 유저별 응답 수 카운트
            user_response_counts[response.userId] += 1

    # 1. Wasserstein 카테고리별 평균, 분산
    wasserstein_by_category = []

    for category, scores in category_wasserstein_data.items():
        if scores:
            mean_score = float(np.mean(scores))
            variance_score = float(np.var(scores))

            wasserstein_by_category.append(
                schemas.CategoryWassersteinStats(
                    category=category,
                    mean=mean_score,
                    variance=variance_score
                )
            )

    # 2. User response 평균, 분산 (user 수 20으로 고정)
    FIXED_USER_COUNT = 20
    total_responses = len(responses)

    # 각 유저별 응답 수 리스트
    user_response_list = list(user_response_counts.values())

    # 만약 실제 유저 수가 20보다 적으면 0으로 채움
    while len(user_response_list) < FIXED_USER_COUNT:
        user_response_list.append(0)

    # 만약 실제 유저 수가 20보다 많으면 상위 20명만 사용
    user_response_list = user_response_list[:FIXED_USER_COUNT]

    mean_responses_per_user = float(np.mean(user_response_list))
    variance_responses_per_user = float(np.var(user_response_list))

    user_response_stats = schemas.UserResponseStats(
        mean_responses_per_user=mean_responses_per_user,
        variance_responses_per_user=variance_responses_per_user
    )

    # 3. 카테고리별 응답 분포
    response_distribution_by_category = [
        schemas.CategoryResponseDistribution(
            category=category,
            response_count=count
        )
        for category, count in category_response_counts.items()
    ]

    return schemas.VarianceAnalysisResponse(
        wasserstein_by_category=wasserstein_by_category,
        user_response_stats=user_response_stats,
        response_distribution_by_category=response_distribution_by_category
    )

