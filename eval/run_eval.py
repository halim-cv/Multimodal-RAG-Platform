"""
eval/run_eval.py

CLI evaluation runner for the Multimodal RAG Platform.

Usage:
    python eval/run_eval.py --session SESSION_ID [--top-k 5]

Output:
    - Richly formatted table in terminal
    - eval/reports/eval_{session_id}_{timestamp}.json
"""

import sys
import json
import time
import asyncio
import argparse
import datetime
from pathlib import Path

# Bootstrap path so we can import backend services
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "Text-encoding" / "model" / "code"))

from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich         import box

from eval.metrics      import compute_all
from eval.faithfulness import faithfulness_score_async

# ─────────────────────────────────────────────────
# Lazy import of backend services (models loaded on demand)
# ─────────────────────────────────────────────────
console = Console()


async def run_evaluation(session_id: str, top_k: int = 5) -> dict:
    from backend.services import retrieval_service, generation_service

    benchmark_path = _PROJECT_ROOT / "eval" / "benchmark_dataset.json"
    if not benchmark_path.exists():
        console.print("[red]benchmark_dataset.json not found![/red]")
        sys.exit(1)

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    console.print(Panel(
        f"[bold cyan]Multimodal RAG Evaluation[/bold cyan]\n"
        f"Session : [yellow]{session_id}[/yellow]\n"
        f"Queries : {len(benchmark)}\n"
        f"Top-K   : {top_k}",
        title="🔬 Eval Runner",
    ))

    per_query_results = []
    latencies         = []

    for item in benchmark:
        query         = item["query"]
        relevant_srcs = item.get("relevant_sources", [])

        # ── Retrieve ──────────────────────────────
        t0     = time.time()
        try:
            chunks = retrieval_service.retrieve(session_id, query, top_k=top_k)
        except Exception as exc:
            console.print(f"[yellow]  Skipping '{query[:40]}…': {exc}[/yellow]")
            continue
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        retrieved_srcs = [c["source"] for c in chunks]
        ir_metrics     = compute_all(retrieved_srcs, relevant_srcs, top_k)

        # ── Generate answer ───────────────────────
        answer = ""
        faith  = {"score": 0.5, "reason": "n/a"}
        try:
            answer = await generation_service.generate_answer(query, chunks)
            faith  = await faithfulness_score_async(answer, chunks)
        except Exception as exc:
            console.print(f"[yellow]  Generation failed: {exc}[/yellow]")

        result = {
            "query":          query,
            "retrieved":      retrieved_srcs,
            "relevant":       relevant_srcs,
            "mrr":            ir_metrics["mrr"],
            "hit":            ir_metrics["hit"],
            "precision":      ir_metrics["precision"],
            "faithfulness":   faith["score"],
            "faith_reason":   faith["reason"],
            "latency_ms":     round(latency_ms, 1),
            "answer_snippet": answer[:120],
        }
        per_query_results.append(result)

    if not per_query_results:
        console.print("[red]No results to report.[/red]")
        return {}

    n           = len(per_query_results)
    avg_mrr     = sum(r["mrr"]          for r in per_query_results) / n
    avg_hit     = sum(r["hit"]          for r in per_query_results) / n
    avg_prec    = sum(r["precision"]    for r in per_query_results) / n
    avg_faith   = sum(r["faithfulness"] for r in per_query_results) / n
    avg_lat     = sum(r["latency_ms"]   for r in per_query_results) / n

    # ── Summary table ──────────────────────────────────────
    table = Table(title="📊 Evaluation Results", box=box.ROUNDED, style="cyan")
    table.add_column("Metric",        style="bold white",  width=20)
    table.add_column("Score",         style="bold yellow", width=12, justify="center")
    table.add_column("Interpretation",style="dim white",   width=40)

    def _color(val: float, good: float, ok: float) -> str:
        if val >= good: return f"[green]{val:.4f}[/green]"
        if val >= ok:   return f"[yellow]{val:.4f}[/yellow]"
        return            f"[red]{val:.4f}[/red]"

    table.add_row("MRR@K",        _color(avg_mrr,   0.6, 0.4), "Correct answer ranked near top?")
    table.add_row("Hit@K",        _color(avg_hit,   0.75, 0.5), "Any relevant result in top-K?")
    table.add_row("Precision@K",  _color(avg_prec,  0.4,  0.2), "Fraction of results that are relevant")
    table.add_row("Faithfulness", _color(avg_faith, 0.7,  0.5), "Answer grounded in context? (LLM judge)")
    table.add_row("Avg Latency",  f"[cyan]{avg_lat:.1f} ms[/cyan]", "FAISS retrieval time per query")
    table.add_row("Queries",      f"{n}", "Evaluated queries")

    console.print(table)

    # ── Per-query detail ───────────────────────────────────
    detail_table = Table(title="Per-Query Detail", box=box.SIMPLE, style="dim")
    detail_table.add_column("Query",        max_width=35)
    detail_table.add_column("MRR",  width=6, justify="right")
    detail_table.add_column("Hit",  width=5, justify="right")
    detail_table.add_column("Faith",width=6, justify="right")
    detail_table.add_column("ms",   width=7, justify="right")

    for r in per_query_results:
        detail_table.add_row(
            r["query"][:35],
            f"{r['mrr']:.2f}",
            f"{r['hit']:.0f}",
            f"{r['faithfulness']:.2f}",
            f"{r['latency_ms']:.0f}",
        )
    console.print(detail_table)

    # ── Save JSON report ───────────────────────────────────
    report_dir  = _PROJECT_ROOT / "eval" / "reports"
    report_dir.mkdir(exist_ok=True)
    ts          = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"eval_{session_id}_{ts}.json"

    report = {
        "session_id":    session_id,
        "timestamp":     ts,
        "top_k":         top_k,
        "summary": {
            "mrr_at_k":       round(avg_mrr,   4),
            "hit_at_k":       round(avg_hit,   4),
            "precision_at_k": round(avg_prec,  4),
            "faithfulness":   round(avg_faith, 4),
            "avg_latency_ms": round(avg_lat,   1),
            "num_queries":    n,
        },
        "per_query": per_query_results,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"\n[green]✅ Report saved → {report_path}[/green]\n")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval quality on a session."
    )
    parser.add_argument("--session", required=True, help="Session ID to evaluate")
    parser.add_argument("--top-k",   type=int, default=5, help="Top-K for retrieval")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.session, args.top_k))


if __name__ == "__main__":
    main()
