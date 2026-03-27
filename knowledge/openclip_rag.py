from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import open_clip
import torch
from PIL import Image


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class SearchHit:
    rank: int
    score: float
    image_path: str
    image_name: str


class OpenClipRAGIndex:
    def __init__(
        self,
        image_dir: Path,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()

        self.image_paths: list[Path] = []
        self.image_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return vectors / norms

    def _list_images(self) -> list[Path]:
        if not self.image_dir.exists():
            return []
        files = [p for p in self.image_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
        files.sort()
        return files

    def _encode_images(self, image_paths: Iterable[Path]) -> np.ndarray:
        tensors = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                tensors.append(self._preprocess(img.convert("RGB")))

        if not tensors:
            return np.empty((0, 0), dtype=np.float32)

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = self._model.encode_image(batch)
        vectors = features.detach().cpu().float().numpy()
        return self._normalize(vectors)

    def _encode_text(self, text: str) -> np.ndarray:
        tokens = self._tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
        vector = features.detach().cpu().float().numpy()
        vector = self._normalize(vector)
        return vector[0]

    def _encode_single_image(self, image: Image.Image) -> np.ndarray:
        tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
        vector = features.detach().cpu().float().numpy()
        vector = self._normalize(vector)
        return vector[0]

    def build_index(self) -> int:
        self.image_paths = self._list_images()
        self.image_matrix = self._encode_images(self.image_paths)
        return len(self.image_paths)

    def build_index_from_paths(self, image_paths: Iterable[Path]) -> int:
        self.image_paths = [Path(p) for p in image_paths]
        self.image_matrix = self._encode_images(self.image_paths)
        return len(self.image_paths)

    def _top_hits(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        if self.image_matrix.size == 0:
            return []

        sims = self.image_matrix @ query_vector
        k = min(top_k, len(self.image_paths))
        order = np.argsort(-sims)[:k]

        hits: list[SearchHit] = []
        for rank, idx in enumerate(order, start=1):
            path = self.image_paths[int(idx)]
            hits.append(
                SearchHit(
                    rank=rank,
                    score=float(sims[int(idx)]),
                    image_path=str(path),
                    image_name=path.name,
                )
            )
        return hits

    def search_text(self, query: str, top_k: int = 3) -> list[SearchHit]:
        vector = self._encode_text(query)
        return self._top_hits(vector, top_k)

    def search_image_path(self, image_path: Path, top_k: int = 3) -> list[SearchHit]:
        with Image.open(image_path) as img:
            vector = self._encode_single_image(img)
        return self._top_hits(vector, top_k)

    def search_image_pil(self, image: Image.Image, top_k: int = 3) -> list[SearchHit]:
        vector = self._encode_single_image(image)
        return self._top_hits(vector, top_k)

    def search_mixed(
        self,
        query_text: str,
        image: Image.Image,
        top_k: int = 3,
        text_weight: float = 0.6,
    ) -> list[SearchHit]:
        text_weight = max(0.0, min(1.0, text_weight))
        image_weight = 1.0 - text_weight

        t = self._encode_text(query_text)
        i = self._encode_single_image(image)
        mixed = text_weight * t + image_weight * i
        mixed_norm = np.linalg.norm(mixed)
        if mixed_norm > 0:
            mixed = mixed / mixed_norm
        return self._top_hits(mixed.astype(np.float32), top_k)
