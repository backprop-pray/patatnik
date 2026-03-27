from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image

from openclip_rag import OpenClipRAGIndex


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = Path(os.getenv("KNOWLEDGE_IMAGE_DIR", BASE_DIR / "test_images"))
MODEL_NAME = os.getenv("OPENCLIP_MODEL", "ViT-B-32")
PRETRAINED = os.getenv("OPENCLIP_PRETRAINED", "laion2b_s34b_b79k")

index: OpenClipRAGIndex | None = None


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)


class SearchHitResponse(BaseModel):
    rank: int
    score: float
    image_path: str
    image_name: str


class SearchResponse(BaseModel):
    total_indexed: int
    hits: list[SearchHitResponse]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global index
    index = OpenClipRAGIndex(
        image_dir=IMAGE_DIR,
        model_name=MODEL_NAME,
        pretrained=PRETRAINED,
    )
    index.build_index()
    yield


app = FastAPI(title="Knowledge OpenCLIP RAG", lifespan=lifespan)


def get_index() -> OpenClipRAGIndex:
    if index is None:
        raise HTTPException(status_code=500, detail="Index is not initialized")
    return index


def to_response(rag: OpenClipRAGIndex, hits) -> SearchResponse:
    return SearchResponse(
        total_indexed=len(rag.image_paths),
        hits=[
            SearchHitResponse(
                rank=h.rank,
                score=h.score,
                image_path=h.image_path,
                image_name=h.image_name,
            )
            for h in hits
        ],
    )

class MobileRecommendationRequest(BaseModel):
    plant_id: int
    image_url: str

class MobileRecommendationResponse(BaseModel):
    plant_id: int
    disease: str
    text: str

@app.post("/mobile/recommendation", response_model=MobileRecommendationResponse)
async def mobile_recommendation(payload: MobileRecommendationRequest) -> MobileRecommendationResponse:
    rag = get_index()

    try:
        async with httpx.AsyncClient() as client:
            img_resp = await client.get(payload.image_url, timeout=15.0)
            pil_image = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not download image: {e}")

    hits = rag.search_image_pil(pil_image, top_k=1)

    if not hits:
        raise HTTPException(status_code=500, detail="No results from index")

    top = hits[0]
    disease = top.image_name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
    text = f"Detected: {disease}. Score: {top.score:.2f}."

    return MobileRecommendationResponse(
        plant_id=payload.plant_id,
        disease=disease[:30],
        text=text[:50]
    )

@app.get("/health")
def health() -> dict[str, str | int]:
    rag = get_index()
    return {
        "status": "ok",
        "indexed_images": len(rag.image_paths),
        "image_dir": str(rag.image_dir),
        "model": rag.model_name,
        "pretrained": rag.pretrained,
        "device": rag.device,
    }


@app.get("/images")
def list_images() -> dict[str, list[str] | int]:
    rag = get_index()
    return {
        "count": len(rag.image_paths),
        "images": [str(p) for p in rag.image_paths],
    }


@app.post("/reindex")
def reindex() -> dict[str, int]:
    rag = get_index()
    count = rag.build_index()
    return {"indexed_images": count}


@app.post("/search/text", response_model=SearchResponse)
def search_text(payload: TextSearchRequest) -> SearchResponse:
    rag = get_index()
    hits = rag.search_text(payload.query, top_k=payload.top_k)
    return to_response(rag, hits)


@app.post("/search/image", response_model=SearchResponse)
async def search_image(
    image: UploadFile = File(...),
    top_k: int = Form(default=3),
) -> SearchResponse:
    rag = get_index()
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")

    try:
        pil_image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}") from exc

    hits = rag.search_image_pil(pil_image, top_k=top_k)
    return to_response(rag, hits)


@app.post("/search/mixed", response_model=SearchResponse)
async def search_mixed(
    query: str = Form(...),
    image: UploadFile = File(...),
    top_k: int = Form(default=3),
    text_weight: float = Form(default=0.6),
) -> SearchResponse:
    rag = get_index()
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")

    try:
        pil_image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}") from exc

    hits = rag.search_mixed(query_text=query, image=pil_image, top_k=top_k, text_weight=text_weight)
    return to_response(rag, hits)
