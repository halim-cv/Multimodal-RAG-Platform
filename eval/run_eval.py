"""
eval/run_eval.py

CLI evaluation runner for the Multimodal RAG Platform.

Usage:
    python eval/run_eval.py --session SESSION_ID [--top-k 5] [--no-mlflow]

Output:
    - Richly formatted table in terminal (via rich)
    - eval/reports/eval_{session_id}_{timestamp}.json
    - MLflow run logged to local mlruns/ (or MLFLOW_TRACKING_URI if set)
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

console = Console()


# ─────────────────────────────────────────────────
# Keyword-overlap score
# ─────────────────────────────────────────────────
def keyword_score(answer: str, expected_keywords: list[str]) -> float:
    """
    Fraction of expected_keywords that appear (case-insensitive) in the answer.
    Returns 1.0 when the list is empty (not a meaningful test).
    """
    if not expected_keywords:
        return 1.0
    ans_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in ans_lower)
    return hits / len(expected_keywords)


# ─────────────────────────────────────────────────
# Main evaluation coroutine
# ─────────────────────────────────────────────────
async def run_evaluation(
    session_id:    str,
    top_k:         int = 5,
    use_mlflow:    bool = True,
) -> dict:
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
        f"Top-K   : {top_k}\n"
        f"MLflow  : {'✅ enabled' if use_mlflow else '⛔ disabled'}",
        title="🔬 Eval Runner",
    ))

    per_query_results = []
    latencies         = []

    for item in benchmark:
        query         = item["query"]
        relevant_srcs = item.get("relevant_sources", [])
        exp_keywords  = item.get("expected_keywords", [])

        console.print(f"\n[cyan]▶ Query:[/cyan] {query[:70]}")

        # ── Retrieve ──────────────────────────────────────────────────────
        t0 = time.time()
        try:
            chunks = retrieval_service.retrieve(session_id, query, top_k=top_k)
        except Exception as exc:
            console.print(f"[yellow]  ⚠ Skipping (retrieve failed): {exc}[/yellow]")
            continue
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        retrieved_srcs = [c["source"] for c in chunks]
        ir_metrics     = compute_all(retrieved_srcs, relevant_srcs, top_k)

        # ── Generate answer ───────────────────────────────────────────────
        answer = ""
        faith  = {"score": 0.5, "reason": "n/a"}
        kw_sc  = 1.0
        try:
            answer = await generation_service.generate_answer(query, chunks)
            faith  = await faithfulness_score_async(answer, chunks)
            kw_sc  = keyword_score(answer, exp_keywords)
        except Exception as exc:
            console.print(f"[yellow]  ⚠ Generation failed: {exc}[/yellow]")

        result = {
            "query":          query,
            "retrieved":      retrieved_srcs,
            "relevant":       relevant_srcs,
            "mrr":            ir_metrics["mrr"],
            "hit":            ir_metrics["hit"],
            "precision":      ir_metrics["precision"],
            "recall":         ir_metrics["recall"],
            "ap":             ir_metrics["ap"],
            "faithfulness":   faith["score"],
            "faith_reason":   faith["reason"],
            "keyword_score":  kw_sc,
            "latency_ms":     round(latency_ms, 1),
            "answer_snippet": answer[:200],
        }
        per_query_results.append(result)

        console.print(
            f"  MRR={result['mrr']:.2f}  Hit={result['hit']:.0f}  "
            f"Faith={result['faithfulness']:.2f}  KW={result['keyword_score']:.2f}  "
            f"Lat={result['latency_ms']:.0f}ms"
        )

        # ── Per-query MLflow log ──────────────────────────────────────────
        if use_mlflow:
            try:
                from mlflow_tracking.tracker import tracker
                tracker.log_query(
                    session_id, query, chunks,
                    {
                        "mrr":          ir_metrics["mrr"],
                        "hit":          ir_metrics["hit"],
                        "precision":    ir_metrics["precision"],
                        "recall":       ir_metrics["recall"],
                        "faithfulness": faith["score"],
                        "keyword_score":kw_sc,
                        "latency_ms":   latency_ms,
                    },
                    run_name=f"q_{item.get('id','?')}_{session_id[:6]}",
                )
            except Exception as exc:
                console.print(f"[dim]  [MLflow] per-query log skipped: {exc}[/dim]")

    if not per_query_results:
        console.print("[red]No results to report.[/red]")
        return {}

    # ── Aggregate metrics ─────────────────────────────────────────────────
    n = len(per_query_results)
    def avg(key): return sum(r[key] for r in per_query_results) / n

    avg_mrr   = avg("mrr")
    avg_hit   = avg("hit")
    avg_prec  = avg("precision")
    avg_rec   = avg("recall")
    avg_faith = avg("faithfulness")
    avg_kw    = avg("keyword_score")
    avg_lat   = avg("latency_ms")

    # ── Rich summary table ────────────────────────────────────────────────
    table = Table(title="📊 Evaluation Results", box=box.ROUNDED, style="cyan")
    table.add_column("Metric",         style="bold white",  width=22)
    table.add_column("Score",          style="bold yellow", width=12, justify="center")
    table.add_column("Interpretation", style="dim white",   width=44)

    def _color(val: float, good: float, ok: float) -> str:
        if val >= good: return f"[green]{val:.4f}[/green]"
        if val >= ok:   return f"[yellow]{val:.4f}[/yellow]"
        return             f"[red]{val:.4f}[/red]"

    table.add_row("MRR@K",        _color(avg_mrr,   0.6, 0.4),  "First relevant result ranked near top?")
    table.add_row("Hit@K",        _color(avg_hit,   0.75,0.5),  "Any relevant result in top-K?")
    table.add_row("Precision@K",  _color(avg_prec,  0.4, 0.2),  "Fraction of top-K that are relevant")
    table.add_row("Recall@K",     _color(avg_rec,   0.5, 0.3),  "Fraction of relevant items retrieved")
    table.add_row("Faithfulness", _color(avg_faith, 0.7, 0.5),  "Answer grounded in context? (LLM judge)")
    table.add_row("Keyword Score",_color(avg_kw,    0.7, 0.5),  "Expected keywords present in answer")
    table.add_row("Avg Latency",  f"[cyan]{avg_lat:.1f} ms[/cyan]", "FAISS retrieval time per query")
    table.add_row("Queries",      f"{n}",                        "Total evaluated queries")
    console.print(table)

    # ── Per-query detail table ────────────────────────────────────────────
    detail = Table(title="Per-Query Detail", box=box.SIMPLE, style="dim")
    detail.add_column("ID",    width=4)
    detail.add_column("Query", max_width=36)
    detail.add_column("MRR",   width=6,  justify="right")
    detail.add_column("Hit",   width=5,  justify="right")
    detail.add_column("Rec",   width=5,  justify="right")
    detail.add_column("Faith", width=6,  justify="right")
    detail.add_column("KW",    width=5,  justify="right")
    detail.add_column("ms",    width=6,  justify="right")

    for i, r in enumerate(per_query_results, 1):
        detail.add_row(
            str(i),
            r["query"][:36],
            f"{r['mrr']:.2f}",
            f"{r['hit']:.0f}",
            f"{r['recall']:.2f}",
            f"{r['faithfulness']:.2f}",
            f"{r['keyword_score']:.2f}",
            f"{r['latency_ms']:.0f}",
        )
    console.print(detail)

    # ── Build full report ─────────────────────────────────────────────────
    ts         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = _PROJECT_ROOT / "eval" / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"eval_{session_id}_{ts}.json"

    report = {
        "session_id": session_id,
        "timestamp":  ts,
        "top_k":      top_k,
        "summary": {
            "mrr_at_k":       round(avg_mrr,   4),
            "hit_at_k":       round(avg_hit,   4),
            "precision_at_k": round(avg_prec,  4),
            "recall_at_k":    round(avg_rec,   4),
            "faithfulness":   round(avg_faith, 4),
            "keyword_score":  round(avg_kw,    4),
            "avg_latency_ms": round(avg_lat,   1),
            "num_queries":    n,
        },
        "per_query": per_query_results,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    console.print(f"\n[green]✅ Report saved → {report_path}[/green]")

    # ── Log full eval report to MLflow ────────────────────────────────────
    if use_mlflow:
        try:
            from mlflow_tracking.tracker import tracker
            run_id = tracker.log_eval_report(report)
            if run_id:
                console.print(f"[green]✅ MLflow eval run logged → run_id: {run_id}[/green]")
        except Exception as exc:
            console.print(f"[dim][MLflow] eval report log skipped: {exc}[/dim]")

    return report


# ─────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval & generation quality on a session."
    )
    parser.add_argument("--session",   required=True,    help="Session ID to evaluate")
    parser.add_argument("--top-k",     type=int,default=5,help="Top-K for retrieval")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.session, args.top_k, not args.no_mlflow))


if __name__ == "__main__":
    main()
