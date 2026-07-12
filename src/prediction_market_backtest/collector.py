"""Reconnect-safe asynchronous CLOB order-book collector."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Awaitable, Callable, Iterable, Protocol

from .orderbook import OrderBook, SequenceGap


class CollectorSink(Protocol):
    async def store_event(self, token_id: str, received_at: datetime, message: dict[str, Any], book_hash: str | None) -> None: ...
    async def store_snapshot(self, book: OrderBook, received_at: datetime, reason: str) -> None: ...


@dataclass(slots=True)
class CollectorHealth:
    connected: bool = False
    disconnects: int = 0
    reconnects: int = 0
    sequence_gaps: int = 0
    reconciliation_failures: int = 0
    snapshot_failures: int = 0
    messages: int = 0
    snapshots: int = 0
    last_message_by_token: dict[str, datetime] = field(default_factory=dict)
    last_error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True)


class MemorySink:
    def __init__(self) -> None: self.events: list[tuple[Any, ...]] = []; self.snapshots: list[tuple[Any, ...]] = []
    async def store_event(self, token_id: str, received_at: datetime, message: dict[str, Any], book_hash: str | None) -> None:
        self.events.append((token_id, received_at, message, book_hash))
    async def store_snapshot(self, book: OrderBook, received_at: datetime, reason: str) -> None:
        self.snapshots.append((book.token_id, book.book_hash, received_at, reason))


class JsonlSink:
    """Append-only local acceptance sink; snapshots are never updated in place."""
    def __init__(self,directory:str|Path)->None:
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.event_path=self.directory/"raw_events.jsonl";self.snapshot_path=self.directory/"snapshots.jsonl"
    async def store_event(self,token_id:str,received_at:datetime,message:dict[str,Any],book_hash:str|None)->None:
        self._append(self.event_path,{"token_id":token_id,"received_at":received_at.isoformat(),"book_hash":book_hash,"message":message})
    async def store_snapshot(self,book:OrderBook,received_at:datetime,reason:str)->None:
        self._append(self.snapshot_path,{"token_id":book.token_id,"received_at":received_at.isoformat(),"reason":reason,"book_hash":book.book_hash,"valid":book.valid,"sequence":book.sequence,"bids":[[str(k),str(v)] for k,v in sorted(book.bids.items(),reverse=True)],"asks":[[str(k),str(v)] for k,v in sorted(book.asks.items())]})
    @staticmethod
    def _append(path:Path,payload:dict[str,Any])->None:
        with path.open("a",encoding="utf-8") as handle:handle.write(json.dumps(payload,sort_keys=True,default=str)+"\n")


class ClobCollector:
    def __init__(self, token_ids: Iterable[str], sink: CollectorSink, *, rest_book: Callable[[str], Awaitable[dict[str, Any]]] | None = None) -> None:
        self.books = {str(token): OrderBook(str(token)) for token in token_ids}
        if not self.books: raise ValueError("at least one token subscription is required")
        self.sink, self.rest_book, self.health = sink, rest_book, CollectorHealth()

    async def process(self, message: dict[str, Any], received_at: datetime | None = None) -> None:
        received_at = received_at or datetime.now(timezone.utc)
        token = str(message.get("asset_id") or message.get("token_id") or "")
        if token not in self.books: return
        book = self.books[token]; event_type = str(message.get("event_type") or message.get("type") or "")
        sequence = _integer(message.get("sequence") or message.get("seq"))
        before = (book.best_bid, book.best_ask)
        try:
            if event_type in {"book", "snapshot"}:
                book.snapshot(_parse_levels(message.get("bids", [])), _parse_levels(message.get("asks", [])), sequence)
            elif event_type in {"price_change", "change", "delta"}:
                for change in message.get("price_changes", message.get("changes", [message])):
                    book.update(str(change.get("side")), change.get("price"), change.get("size"), sequence)
                    sequence = None  # one envelope sequence may contain multiple changes
            else:
                await self.sink.store_event(token, received_at, message, book.book_hash if book.valid else None); return
        except SequenceGap as exc:
            self.health.sequence_gaps += 1; self.health.last_error = str(exc); book.valid = False
            await self.sink.store_event(token, received_at, message, None)
            if self.rest_book: await self.reconcile(token, "sequence_gap")
            return
        self.health.messages += 1; self.health.last_message_by_token[token] = received_at
        await self.sink.store_event(token, received_at, message, book.book_hash if book.valid else None)
        if event_type in {"book", "snapshot"} or before != (book.best_bid, book.best_ask):
            await self._snapshot(book, received_at, "initial_or_best_price_change")

    async def reconcile(self, token_id: str, reason: str = "scheduled_reconciliation") -> bool:
        if not self.rest_book: raise RuntimeError("REST reconciliation callback is required")
        try:
            payload = await self.rest_book(token_id); book = self.books[token_id]
            book.snapshot(_parse_levels(payload.get("bids", [])), _parse_levels(payload.get("asks", [])), _integer(payload.get("sequence")))
            await self._snapshot(book, datetime.now(timezone.utc), reason); return book.valid
        except Exception as exc:
            self.health.reconciliation_failures += 1; self.health.last_error = str(exc); self.books[token_id].valid = False; return False

    async def _snapshot(self, book: OrderBook, received_at: datetime, reason: str) -> None:
        try: await self.sink.store_snapshot(book, received_at, reason); self.health.snapshots += 1
        except Exception as exc: self.health.snapshot_failures += 1; self.health.last_error = str(exc); raise

    async def run(self, websocket_url: str, *, reconnect_delay: float = 1.0, controlled_disconnect_after: float | None = None,
                  snapshot_interval: float = 60.0) -> None:
        try: import websockets
        except ImportError as exc: raise RuntimeError("capture-books requires websockets>=15,<16") from exc
        first = True
        snapshot_task=asyncio.create_task(self._snapshot_periodically(snapshot_interval))
        try:
          while True:
            try:
                async with websockets.connect(websocket_url, ping_interval=20, ping_timeout=20) as socket:
                    self.health.connected = True
                    if not first: self.health.reconnects += 1
                    first = False
                    await socket.send(json.dumps({"assets_ids": list(self.books), "type": "market"}))
                    for token in self.books:
                        if self.rest_book: await self.reconcile(token, "connect_or_reconnect")
                    disconnect_task=None
                    if controlled_disconnect_after is not None and self.health.reconnects==0:
                        async def disconnect()->None:
                            await asyncio.sleep(controlled_disconnect_after);self.health.disconnects+=1;self.health.last_error="controlled_disconnect";await socket.close(code=1000,reason="controlled_acceptance_test")
                        disconnect_task=asyncio.create_task(disconnect())
                    async for raw in socket:
                        payload = json.loads(raw)
                        for message in payload if isinstance(payload, list) else [payload]: await self.process(message)
                    if disconnect_task:disconnect_task.cancel()
            except asyncio.CancelledError: self.health.connected = False; raise
            except Exception as exc:
                self.health.connected = False; self.health.disconnects += 1; self.health.last_error = str(exc)
                for book in self.books.values(): book.valid = False
                await asyncio.sleep(reconnect_delay)
        finally:
            snapshot_task.cancel()
            try:await snapshot_task
            except asyncio.CancelledError:pass

    async def _snapshot_periodically(self,interval:float)->None:
        while True:
            await asyncio.sleep(interval);now=datetime.now(timezone.utc)
            for book in self.books.values():
                if book.valid:await self._snapshot(book,now,"scheduled_interval")


def _parse_levels(levels: Iterable[Any]) -> list[tuple[Any, Any]]:
    return [(level.get("price"), level.get("size")) if isinstance(level, dict) else (level[0], level[1]) for level in levels]
def _integer(value: Any) -> int | None:
    try: return int(value) if value is not None else None
    except (TypeError, ValueError): return None
