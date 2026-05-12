# ACE-Router — Inference & Evaluation Code

This directory contains the inference and evaluation code for the paper
**ACE-Router: Generalizing History-Aware Routing from MCP Tools to the Agent Web**
([arXiv:2601.08276](https://arxiv.org/abs/2601.08276), ACL 2026 Main).

> Training pipeline, data generation, and model weights are **not** included.
> Deploy your own router model via vLLM / SGLang and point the YAML config at it.

---

## Repository Layout

```
mcpuniverse/
├── agent/
│   ├── base.py              # BaseAgent — MCP manager, LLM, tracer, callbacks
│   ├── react.py             # ReAct baseline agent
│   ├── dynamic_react.py     # Light Routing Agent (LRA) — core contribution
│   ├── router/              # ToolRouter + LLM / embedding backends
│   │   ├── tool_router.py
│   │   ├── config.py
│   │   ├── history.py       # History formatting (v1 / v2)
│   │   ├── response_parser.py
│   │   └── backends/        # llm.py, embedding.py
│   └── configs/             # Jinja2 prompt templates
├── benchmark/
│   ├── runner.py            # BenchmarkRunner
│   ├── task.py              # Task definition
│   ├── mcpuniverse/         # MCP-Universe benchmark (6 domains)
│   │   ├── configs/         # Per-domain YAML configs
│   │   └── tasks/           # Task JSON files
│   └── mcpmark/             # MCPMark benchmark (5 domains)
│       ├── configs/         # Per-domain YAML configs
│       └── tasks/           # Task JSON files
├── llm/                     # LLM backends (OpenAI-compatible + others)
├── mcp/                     # MCP client, server configs, built-in servers
├── evaluator/               # Evaluation functions
├── tracer/                  # Trace collection
└── callbacks/               # Logging callbacks

scripts/
└── run_dynamic_react_smoke.py   # End-to-end smoke test
```

---

## Setup

```bash
conda create -n toolace python=3.11 -y
conda activate toolace
conda install -c conda-forge nodejs -y   # required by MCPMark servers
pip install -e .
```

Copy `.env.example` to `.env` and fill in the API keys for whichever MCP
servers you plan to use.

---

## Quick Start — Smoke Test

```bash
conda activate toolace
mkdir -p log

# MCP-Universe (weather server, single task)
python scripts/run_dynamic_react_smoke.py mcpuniverse

# MCPMark (filesystem server, single task)
python scripts/run_dynamic_react_smoke.py mcpmark
```

Expected output: `Summary: 1/1 tasks passed`

---

## Running a Full Benchmark

Point the runner at any YAML config under `mcpuniverse/benchmark/`:

```bash
python - <<'EOF'
import asyncio
from mcpuniverse.benchmark.runner import BenchmarkRunner
from mcpuniverse.benchmark.report import BenchmarkReport
from mcpuniverse.tracer.collectors import FileCollector

async def main():
    runner = BenchmarkRunner("mcpuniverse/configs/location_navigation.yaml")
    collector = FileCollector(log_file="log/trace.log")
    results = await runner.run(trace_collector=collector)
    BenchmarkReport(runner, trace_collector=collector).dump()

asyncio.run(main())
EOF
```

---

## Enabling ACE-Router (DynamicReAct)

Change `type: react` → `type: dynamic_react` and add a `router:` block:

```yaml
kind: agent
spec:
  name: my-agent
  type: dynamic_react
  config:
    llm: reasoning-llm
    max_iterations: 20
    servers:
      - name: github
      - name: filesystem
    router:
      mode: llm          # or "embedding"
      max_tools: 5
      enable_history: true
      history_version: v1
      llm:
        type: openai
        config:
          model_name: <your-router-model>
          base_url: "http://localhost:10121/v1"
          api_key: "token"
```

### RouterConfig reference

| Field | Default | Description |
|---|---|---|
| `mode` | `"llm"` | `"llm"` or `"embedding"` |
| `max_tools` | `0` | Max tools returned (0 = unlimited) |
| `enable_history` | `true` | Feed conversation history to router |
| `history_version` | `"v1"` | `v1`: execute-tool only · `v2`: route + execute-tool |
| `shuffle_tools` | `false` | Shuffle tool order to reduce position bias |
| `embedding_model` | `"local"` | `local` · `openai` · `qwen3` · `bm25` · `contriever` |

---

## Benchmarks

### MCP-Universe (6 domains)

| Domain | Config |
|---|---|
| Location & Navigation | `mcpuniverse/configs/location_navigation.yaml` |
| Repository Management | `mcpuniverse/configs/repository_management.yaml` |
| Financial Analysis | `mcpuniverse/configs/financial_analysis.yaml` |
| 3D Design | `mcpuniverse/configs/3d_design.yaml` |
| Browser Automation | `mcpuniverse/configs/browser_automation.yaml` |
| Web Search | `mcpuniverse/configs/web_search.yaml` |

### MCPMark (5 domains)

| Domain | Config |
|---|---|
| Filesystem | `mcpmark/configs/mcpmark_filesystem.yaml` |
| GitHub | `mcpmark/configs/mcpmark_github.yaml` |
| Notion | `mcpmark/configs/mcpmark_notion.yaml` |
| PostgreSQL | `mcpmark/configs/mcpmark_postgres.yaml` |
| Playwright | `mcpmark/configs/mcpmark_playwright.yaml` |

---

## Citation

```bibtex
@misc{acerouter2026,
  title={ACE-Router: Generalizing History-Aware Routing from MCP Tools to the Agent Web},
  author={Zhiyuan Yao and Zishan Xu and Yifu Guo and Zhiguang Han and Cheng Yang and
          Shuo Zhang and Weinan Zhang and Xingshan Zeng and Weiwen Liu},
  year={2026},
  eprint={2601.08276},
  archivePrefix={arXiv},
}
```
