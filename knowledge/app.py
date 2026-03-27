from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
import psycopg

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image

from openclip_rag import OpenClipRAGIndex


BASE_DIR = Path(__file__).resolve().parent
REFERENCE_CACHE_DIR = Path(os.getenv("KNOWLEDGE_CACHE_DIR", "/tmp/patatnik-knowledge-cache"))
MODEL_NAME = os.getenv("OPENCLIP_MODEL", "ViT-B-32")
PRETRAINED = os.getenv("OPENCLIP_PRETRAINED", "laion2b_s34b_b79k")

index: OpenClipRAGIndex | None = None
has_health_status_column = False


@dataclass
class ReferenceRecommendation:
    disease: str
    recommendation: str
    recommended_action_user_id: int


reference_by_path: dict[str, ReferenceRecommendation] = {}
reference_by_name: dict[str, ReferenceRecommendation] = {}


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


def parse_db_settings() -> dict[str, str | int]:
    spring_url = os.getenv("SPRING_DATASOURCE_URL", "").strip()
    spring_user = os.getenv("SPRING_DATASOURCE_USERNAME", "").strip()
    spring_password = os.getenv("SPRING_DATASOURCE_PASSWORD", "").strip()

    if spring_url.startswith("jdbc:"):
        spring_url = spring_url[5:]

    parsed = urlparse(spring_url)
    if parsed.scheme != "postgresql" or not parsed.hostname or not parsed.path:
        raise RuntimeError(
            "Database configuration missing. Set Spring datasource env vars."
        )
    if not spring_user or not spring_password:
        raise RuntimeError(
            "Database configuration missing. Set SPRING_DATASOURCE_USERNAME and SPRING_DATASOURCE_PASSWORD."
        )

    return {
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "dbname": parsed.path.lstrip("/"),
        "user": spring_user,
        "password": spring_password,
    }


DB_SETTINGS = parse_db_settings()


def open_db_connection() -> psycopg.Connection:
    return psycopg.connect(
        host=DB_SETTINGS["host"],
        port=DB_SETTINGS["port"],
        dbname=DB_SETTINGS["dbname"],
        user=DB_SETTINGS["user"],
        password=DB_SETTINGS["password"],
    )


def normalize_image_path_key(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def normalize_image_name_key(path_or_url: str) -> str:
    parsed = urlparse(path_or_url)
    source = parsed.path if parsed.scheme in {"http", "https"} else path_or_url
    return Path(unquote(source)).name.lower()


def cache_reference_image(image_url: str, processed_plant_id: int) -> Path:
    REFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(image_url)
    suffix = Path(unquote(parsed.path)).suffix or ".png"
    local_path = REFERENCE_CACHE_DIR / f"processed_{processed_plant_id}{suffix}"

    if local_path.exists():
        return local_path

    if parsed.scheme in {"http", "https"}:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            response = client.get(image_url)
            response.raise_for_status()
            local_path.write_bytes(response.content)
        return local_path

    source_path = Path(image_url)
    local_path.write_bytes(source_path.read_bytes())
    return local_path


def load_reference_catalog() -> tuple[list[Path], dict[str, ReferenceRecommendation], dict[str, ReferenceRecommendation]]:
    image_paths: list[Path] = []
    by_path: dict[str, ReferenceRecommendation] = {}
    by_name: dict[str, ReferenceRecommendation] = {}

    query = (
        "SELECT pp.id, p.image_url, COALESCE(pp.disease, ''), pp.recommended_action, pp.recommended_action_user_id "
        "FROM public.processed_plants pp "
        "JOIN public.plants p ON p.id = pp.plant_id "
        "WHERE p.image_url IS NOT NULL AND p.image_url <> '' "
        "AND pp.recommended_action IS NOT NULL AND pp.recommended_action <> '' "
        "ORDER BY pp.id DESC"
    )

    with open_db_connection() as conn, conn.cursor() as cur:
        cur.execute(query)
        for processed_plant_id, image_url, disease, recommendation, source_user_id in cur.fetchall():
            payload = ReferenceRecommendation(
                disease=disease,
                recommendation=recommendation,
                recommended_action_user_id=int(source_user_id),
            )

            cached_path = cache_reference_image(image_url, int(processed_plant_id))
            image_paths.append(cached_path)
            by_path[normalize_image_path_key(str(cached_path))] = payload
            by_name[normalize_image_name_key(str(cached_path))] = payload

    return image_paths, by_path, by_name


def resolve_reference(hit_path: str, hit_name: str) -> ReferenceRecommendation | None:
    path_key = normalize_image_path_key(hit_path)
    if path_key in reference_by_path:
        return reference_by_path[path_key]
    return reference_by_name.get(hit_name.lower())


def detect_health_status_column() -> bool:
    query = (
        "SELECT EXISTS ("
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = 'processed_plants' AND column_name = 'health_status'"
        ")"
    )
    with open_db_connection() as conn, conn.cursor() as cur:
        cur.execute(query)
        row = cur.fetchone()
        if not row:
            return False
        return bool(row[0])


def upsert_processed_plant(
    plant_id: int,
    recommendation: ReferenceRecommendation,
) -> None:
    health_status = "DISEASED" if recommendation.disease else "HEALTHY"

    with open_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM public.processed_plants "
            "WHERE plant_id = %s AND disease IS NOT DISTINCT FROM %s "
            "ORDER BY id DESC LIMIT 1",
            (plant_id, recommendation.disease),
        )
        row = cur.fetchone()

        if row:
            if has_health_status_column:
                cur.execute(
                    "UPDATE public.processed_plants "
                    "SET recommended_action = %s, recommended_action_user_id = %s, health_status = %s "
                    "WHERE id = %s",
                    (
                        recommendation.recommendation,
                        recommendation.recommended_action_user_id,
                        health_status,
                        int(row[0]),
                    ),
                )
            else:
                cur.execute(
                    "UPDATE public.processed_plants "
                    "SET recommended_action = %s, recommended_action_user_id = %s "
                    "WHERE id = %s",
                    (
                        recommendation.recommendation,
                        recommendation.recommended_action_user_id,
                        int(row[0]),
                    ),
                )
            conn.commit()
            return

        if has_health_status_column:
            cur.execute(
                "INSERT INTO public.processed_plants "
                "(plant_id, disease, health_status, recommended_action, recommended_action_user_id, created_at) "
                "VALUES (%s, %s, %s, %s, %s, NOW())",
                (
                    plant_id,
                    recommendation.disease,
                    health_status,
                    recommendation.recommendation,
                    recommendation.recommended_action_user_id,
                ),
            )
        else:
            cur.execute(
                "INSERT INTO public.processed_plants "
                "(plant_id, disease, recommended_action, recommended_action_user_id, created_at) "
                "VALUES (%s, %s, %s, %s, NOW())",
                (
                    plant_id,
                    recommendation.disease,
                    recommendation.recommendation,
                    recommendation.recommended_action_user_id,
                ),
            )

        conn.commit()


def refresh_reference_catalog() -> None:
    global reference_by_path
    global reference_by_name

    _, reference_by_path, reference_by_name = load_reference_catalog()


def update_processed_plant_recommendation(
    processed_plant_id: int,
    text: str,
) -> tuple[int, int, str, str]:
    with open_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, plant_id, COALESCE(disease, '') FROM public.processed_plants WHERE id = %s",
            (processed_plant_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="processed_plant not found")

        row_id = int(row[0])
        plant_id = int(row[1])
        disease = row[2]

        if disease:
            cur.execute(
                "UPDATE public.processed_plants SET recommended_action = %s WHERE disease = %s",
                (text, disease),
            )
        else:
            cur.execute(
                "UPDATE public.processed_plants SET recommended_action = %s WHERE id = %s",
                (text, processed_plant_id),
            )

        conn.commit()

    refresh_reference_catalog()
    return row_id, plant_id, disease, text


@asynccontextmanager
async def lifespan(_: FastAPI):
    global has_health_status_column
    global index
    global reference_by_path
    global reference_by_name

    index = OpenClipRAGIndex(
        image_dir=REFERENCE_CACHE_DIR,
        model_name=MODEL_NAME,
        pretrained=PRETRAINED,
    )
    has_health_status_column = detect_health_status_column()
    image_paths, reference_by_path, reference_by_name = load_reference_catalog()
    index.build_index_from_paths(image_paths)
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


class UpdateProcessedPlantRecommendationRequest(BaseModel):
    processed_plant_id: int = Field(ge=1)
    text: str = Field(min_length=1)


class UpdateProcessedPlantRecommendationResponse(BaseModel):
    processed_plant_id: int
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
    resolved = resolve_reference(top.image_path, top.image_name)
    if resolved is None:
        raise HTTPException(status_code=404, detail="No recommendation in processed_plants for top match")

    try:
        upsert_processed_plant(payload.plant_id, resolved)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to persist processed plant: {exc}") from exc

    return MobileRecommendationResponse(
        plant_id=payload.plant_id,
        disease=resolved.disease,
        text=resolved.recommendation,
    )


@app.patch(
    "/processed-plants/recommendation",
    response_model=UpdateProcessedPlantRecommendationResponse,
)
def update_recommendation(
    payload: UpdateProcessedPlantRecommendationRequest,
) -> UpdateProcessedPlantRecommendationResponse:
    processed_plant_id, plant_id, disease, text = update_processed_plant_recommendation(
        payload.processed_plant_id,
        payload.text.strip(),
    )
    return UpdateProcessedPlantRecommendationResponse(
        processed_plant_id=processed_plant_id,
        plant_id=plant_id,
        disease=disease,
        text=text,
    )

@app.get("/health")
def health() -> dict[str, str | int]:
    rag = get_index()
    return {
        "status": "ok",
        "indexed_images": len(rag.image_paths),
        "image_dir": str(REFERENCE_CACHE_DIR),
        "model": rag.model_name,
        "pretrained": rag.pretrained,
        "device": rag.device,
    }


"""
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
"""
