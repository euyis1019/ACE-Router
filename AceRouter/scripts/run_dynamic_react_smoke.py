"""
Runnable smoke test for DynamicReAct on both benchmarks.

Usage::

    export OPENAI_API_KEY=sk-...
    # (Optional) if your router LLM lives elsewhere, edit the YAMLs directly.
    python scripts/run_dynamic_react_smoke.py mcpuniverse
    python scripts/run_dynamic_react_smoke.py mcpmark

Both commands run a single task end-to-end. The first will hit the built-in
weather server; the second will launch the MCPMark filesystem server (npx-based).
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path
from mcpuniverse.benchmark.runner import BenchmarkRunner
from mcpuniverse.benchmark.report import BenchmarkReport
from mcpuniverse.callbacks.handlers.vprint import get_vprint_callbacks
from mcpuniverse.tracer.collectors import FileCollector


_BENCHMARK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "mcpuniverse", "benchmark",
)

CONFIGS = {
    "mcpuniverse": os.path.join(_BENCHMARK_DIR, "mcpuniverse", "configs", "smoke_dynamic_react.yaml"),
    "mcpmark": os.path.join(_BENCHMARK_DIR, "mcpmark", "configs", "mcpmark_filesystem_dynamic_react.yaml"),
}


async def _run(config_path: str, target: str) -> int:
    # Trace log: every LLM / tool / router call's full messages + response ends up here.
    log_path = f"log/{target}_trace.log"
    print(f"Trace log → {log_path}")
    trace_collector = FileCollector(log_file=log_path)
    runner = BenchmarkRunner(config_path)
    results = await runner.run(
        trace_collector=trace_collector,
        callbacks=get_vprint_callbacks(),
    )
    report = BenchmarkReport(runner, trace_collector=trace_collector)
    report.dump()

    # Summarize pass/fail for each task.
    passed = 0
    total = 0
    for br in results:
        for task_name, task_data in br.task_results.items():
            total += 1
            evals = task_data.get("evaluation_results", [])
            ok = all(getattr(e, "passed", False) for e in evals) if evals else False
            mark = "PASS" if ok else "FAIL"
            print(f"[{mark}] {task_name}")
            if ok:
                passed += 1
    print(f"\nSummary: {passed}/{total} tasks passed")
    return 0 if passed == total else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target", choices=list(CONFIGS.keys()),
        help="Which benchmark config to run (mcpuniverse | mcpmark)",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Export it first.", file=sys.stderr)
        return 2

    config_path = CONFIGS[args.target]
    print(f"Running config: {config_path}")
    return asyncio.run(_run(config_path, args.target))


if __name__ == "__main__":
    raise SystemExit(main())
