import base64
import json
import mimetypes
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError


MODEL_NAME = os.getenv("OPENAI_VLM_MODEL", "gpt-5.4")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", "8000000"))

ENABLE_PAST_CASE_RETRIEVAL = False

SYSTEM_PROMPT = """You are a crop disease/pest analytic agent. You will receive location information and an image of a plant/crop with an anomaly detected. Answer very briefly.

Requirements:
1) State what the issue is, if there is one. It may also be a false alarm.
2) If there is a problem, provide up to 3 top suggestions with a 3-level certainty (high, medium, low).
3) For each suggestion, recommend what the farmer should do, including chemicals when appropriate.
4) Keep wording short and practical for farmers.
5) Output valid JSON only, matching the provided schema exactly.
"""

RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "assessment_text",
        "anomaly_status",
        "top_suggestions",
        "monitoring_note",
    ],
    "properties": {
        "assessment_text": {"type": "string"},
        "anomaly_status": {
            "type": "object",
            "additionalProperties": False,
            "required": ["is_real_problem", "short_explanation"],
            "properties": {
                "is_real_problem": {"type": "boolean"},
                "short_explanation": {"type": "string"},
            },
        },
        "top_suggestions": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "rank",
                    "issue_name",
                    "certainty",
                    "recommendation",
                ],
                "properties": {
                    "rank": {"type": "integer", "minimum": 1, "maximum": 3},
                    "issue_name": {"type": "string"},
                    "certainty": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "recommendation": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["action_text", "chemical_name", "urgency"],
                        "properties": {
                            "action_text": {"type": "string"},
                            "chemical_name": {"type": "string"},
                            "urgency": {
                                "type": "string",
                                "enum": ["immediate", "soon", "monitor"],
                            },
                        },
                    },
                },
            },
        },
        "monitoring_note": {"type": "string"},
    },
}


class AnomalyStatus(BaseModel):
    is_real_problem: bool
    short_explanation: str


class Recommendation(BaseModel):
    action_text: str
    chemical_name: str
    urgency: str


class Suggestion(BaseModel):
    rank: int
    issue_name: str
    certainty: str
    recommendation: Recommendation


class RecommendationResponse(BaseModel):
    assessment_text: str
    anomaly_status: AnomalyStatus
    top_suggestions: list[Suggestion]
    monitoring_note: str


def build_retrieval_query(region: str, country: str | None, region_type: str | None) -> None:
    return None


def retrieve_past_problem_description(_query: None) -> str | None:
    return None


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    for output in getattr(response, "output", []):
        for item in getattr(output, "content", []):
            if getattr(item, "type", "") in {"output_text", "text"} and getattr(item, "text", None):
                return item.text

    raise ValueError("No text output returned by OpenAI response")


def _make_context(region: str, country: str | None, region_type: str | None, past_hint: str | None) -> str:
    lines = [f"Region: {region}"]
    if country:
        lines.append(f"Country: {country}")
    if region_type:
        lines.append(f"Region type: {region_type}")
    if past_hint:
        lines.append(f"Past similar case: {past_hint}")
    return "\n".join(lines)


async def analyze_crop_anomaly(
    *,
    client: AsyncOpenAI,
    image_bytes: bytes,
    mime_type: str,
    region: str,
    country: str | None,
    region_type: str | None,
) -> RecommendationResponse:
    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{image_base64}"

    past_hint = None
    if ENABLE_PAST_CASE_RETRIEVAL:
        query = build_retrieval_query(region=region, country=country, region_type=region_type)
        past_hint = retrieve_past_problem_description(query)

    context_block = _make_context(
        region=region,
        country=country,
        region_type=region_type,
        past_hint=past_hint,
    )

    response = await client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"{SYSTEM_PROMPT}\nLocation context:\n{context_block}",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": data_url,
                    }
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "crop_recommendation",
                "schema": RESPONSE_JSON_SCHEMA,
                "strict": True,
            }
        },
        temperature=0.1,
        max_output_tokens=700,
    )

    output_text = _extract_output_text(response)
    parsed = json.loads(output_text)
    return RecommendationResponse.model_validate(parsed)


app = FastAPI(title="Knowledge Recommendation Graph Microservice", version="0.1.0")
openai_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    global openai_client
    if openai_client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
        openai_client = AsyncOpenAI(timeout=OPENAI_TIMEOUT_SECONDS)
    return openai_client


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/recommendation")
async def recommendation(
    image: UploadFile = File(...),
    region: str = Form(...),
    country: str | None = Form(default=None),
    region_type: str | None = Form(default=None),
) -> dict[str, Any]:
    if not region.strip():
        raise HTTPException(status_code=400, detail="region must not be empty")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image is empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"image is too large (max {MAX_IMAGE_BYTES} bytes)")

    mime_type = image.content_type or mimetypes.guess_type(image.filename or "")[0] or "application/octet-stream"
    if not mime_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    request_id = str(uuid.uuid4())
    started_at = time.perf_counter()

    try:
        result = await analyze_crop_anomaly(
            client=get_openai_client(),
            image_bytes=image_bytes,
            mime_type=mime_type,
            region=region,
            country=country,
            region_type=region_type,
        )
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=502, detail=f"model output schema mismatch: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"model call failed: {exc}") from exc

    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)

    return {
        "request_id": request_id,
        "model": MODEL_NAME,
        "latency_ms": latency_ms,
        "recommendation": result.model_dump(),
    }
