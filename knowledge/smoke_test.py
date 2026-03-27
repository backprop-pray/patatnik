from __future__ import annotations

from fastapi.testclient import TestClient

from app import app


def run() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        print("/health", health.status_code, health.json())


if __name__ == "__main__":
    run()
