"""Wall-clock CLOB acceptance runner with explicit diagnostic labeling."""

from __future__ import annotations
import asyncio,json
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

from .acceptance import validate_capture_acceptance
from .clients import ClobClient
from .collector import ClobCollector,JsonlSink


async def run_capture_acceptance(token_ids:list[str],output_dir:str|Path,*,duration_hours:float=24,
    websocket_url:str="wss://ws-subscriptions-clob.polymarket.com/ws/market",diagnostic_only:bool=True,
    controlled_faults:bool=False,fault_delay_seconds:float=60)->dict[str,Any]:
    if not token_ids or any(not isinstance(token,str) or not token.isdigit() for token in token_ids):raise ValueError("string token IDs containing only digits are required")
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True);sink=JsonlSink(output);client=ClobClient("https://clob.polymarket.com")
    async def rest(token:str)->dict[str,Any]:return await asyncio.to_thread(client.book,token)
    collector=ClobCollector(token_ids,sink,rest_book=rest);start=datetime.now(timezone.utc)
    running_manifest={"status":"running","diagnostic_only":diagnostic_only,"box_office_analysis_eligible":False,"start_utc":start.isoformat(),"requested_duration_hours":duration_hours,"subscribed_tokens":token_ids,"controlled_faults":controlled_faults}
    (output/"01_run_manifest.json").write_text(json.dumps(running_manifest,indent=2)+"\n")
    task=asyncio.create_task(collector.run(websocket_url,controlled_disconnect_after=fault_delay_seconds if controlled_faults else None))
    fault_task=None
    if controlled_faults:
        async def induce_gap()->None:
            await asyncio.sleep(fault_delay_seconds*2);token=token_ids[0];book=collector.books[token];book.sequence=1
            await collector.process({"event_type":"price_change","asset_id":token,"sequence":3,"price_changes":[{"side":"BUY","price":"0.01","size":"1"}],"injected_fault":"controlled_sequence_gap"})
        fault_task=asyncio.create_task(induce_gap())
    try:await asyncio.sleep(duration_hours*3600)
    finally:
        task.cancel()
        if fault_task:fault_task.cancel()
        try:await task
        except asyncio.CancelledError:pass
    end=datetime.now(timezone.utc);manifest={"status":"completed_pending_validation","diagnostic_only":diagnostic_only,"box_office_analysis_eligible":not diagnostic_only,"start_utc":start.isoformat(),"end_utc":end.isoformat(),"duration_hours":(end-start).total_seconds()/3600,"subscribed_tokens":token_ids,"messages":collector.health.messages,"snapshots":collector.health.snapshots,"disconnects":collector.health.disconnects,"reconnect_successes":collector.health.reconnects,"sequence_gaps":collector.health.sequence_gaps,"state_invalidations":collector.health.sequence_gaps,"rest_recovery_successes":max(0,collector.health.snapshots-len(token_ids)),"reconciliation_failures":collector.health.reconciliation_failures,"unrecovered_failures":int(any(not book.valid for book in collector.books.values())),"database_write_errors":collector.health.snapshot_failures,"last_error":collector.health.last_error}
    manifest["controlled_faults"]=controlled_faults
    passed,errors=validate_capture_acceptance(manifest);manifest["status"]="passed" if passed else errors[0];manifest["validation_errors"]=errors;(output/"01_run_manifest.json").write_text(json.dumps(manifest,indent=2,default=str)+"\n");(output/"08_health_metrics.json").write_text(collector.health.to_json()+"\n");(output/"summary.md").write_text(f"# CLOB capture acceptance\n\nStatus: **{manifest['status']}**.\n\nDuration: `{manifest['duration_hours']:.4f}` hours.\n\nDiagnostic-only: `{diagnostic_only}`.\n\nErrors: `{errors}`.\n");return manifest
