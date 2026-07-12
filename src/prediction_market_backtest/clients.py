"""Read-only Gamma and CLOB REST clients with bounded retries."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(slots=True)
class JsonClient:
    base_url: str
    timeout: float = 15.0
    retries: int = 3
    opener: Any = urllib.request.urlopen

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None}, doseq=True)
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}" + (f"?{query}" if query else "")
        for attempt in range(self.retries + 1):
            try:
                with self.opener(urllib.request.Request(url, headers={"Accept": "application/json",
                    "User-Agent": "pm-box-office-prediction-market/0.1 (read-only research)"}), timeout=self.timeout) as response:
                    return json.loads(response.read())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                if attempt == self.retries:
                    raise
                time.sleep(min(2 ** attempt, 8))
        raise AssertionError("unreachable")


class GammaClient(JsonClient):
    def events(self, **filters: Any) -> Iterator[dict[str, Any]]:
        limit, offset = int(filters.pop("limit", 100)), 0
        seen: set[str] = set()
        while True:
            page = self.get("events", {**filters, "limit": limit, "offset": offset})
            rows = page if isinstance(page, list) else page.get("data", [])
            for row in rows:
                event_id = str(row.get("id") or row.get("event_id"))
                if event_id not in seen:
                    seen.add(event_id)
                    yield row
            if len(rows) < limit:
                break
            offset += limit


class ClobClient(JsonClient):
    def book(self, token_id: str) -> dict[str, Any]:
        if not isinstance(token_id, str):
            raise TypeError("token IDs must be strings")
        return self.get("book", {"token_id": token_id})

    def price_history(self,token_id:str,*,interval:str="max",fidelity:int=1)->list[dict[str,Any]]:
        if not isinstance(token_id,str):raise TypeError("token IDs must be strings")
        payload=self.get("prices-history",{"market":token_id,"interval":interval,"fidelity":fidelity})
        return payload.get("history",[]) if isinstance(payload,dict) else payload
