from __future__ import annotations

from typing import Any

import httpx

from .config import Settings

BASE_URL = "https://api.blave.org"


class BlaveClient:
    def __init__(self, settings: Settings, timeout: float = 30.0) -> None:
        self.settings = settings
        self.client = httpx.Client(
            base_url=BASE_URL,
            timeout=timeout,
            headers={
                "api-key": settings.blave_api_key,
                "secret-key": settings.blave_secret_key,
            },
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "BlaveClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def credentials_available(self) -> bool:
        return self.settings.has_blave_credentials

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.credentials_available():
            raise RuntimeError("Blave credentials are missing.")
        response = self.client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    def alpha_table(self) -> Any:
        return self._get("/alpha_table")

    def latest_snapshot(self, symbol: str) -> dict[str, Any]:
        payload = self.alpha_table()
        data = payload.get("data") if isinstance(payload, dict) else {}
        symbol_key = symbol.upper()
        row = (data or {}).get(symbol_key) or {}
        if not isinstance(row, dict):
            return {}
        return row
