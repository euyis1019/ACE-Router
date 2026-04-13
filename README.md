<div align="center">

# ACE-Router: Generalizing History-Aware Routing from MCP Tools to the Agent Web

**Accepted to ACL 2026 Main Conference**

[[Paper]](https://arxiv.org/abs/2601.08276)

</div>

## Overview

With the rise of the **Agent Web** and **Model Context Protocol (MCP)**, the agent ecosystem is evolving into an open collaborative network, exponentially increasing accessible tools. However, current architectures face severe scalability and generality bottlenecks.

We propose **ACE-Router**, a pipeline for training **history-aware routers** to empower precise navigation in large-scale ecosystems. By leveraging a dependency-rich candidate graph to synthesize multi-turn trajectories, we effectively train routers with dynamic context understanding to create the plug-and-play **Light Routing Agent**.

<div align="center">
<img src="assets/framework.png" width="95%">
</div>

### Key Highlights

- **Self-Evolutionary Graph Construction** -- Expands and structures the candidate space via mutation and relation modeling.
- **Multi-Agent Simulation** -- Synthesizes interaction trajectories to extract history-aware supervision signals.
- **Light Routing Agent** -- A plug-and-play module that seamlessly integrates the trained router into existing agent pipelines.
- **Cross-domain Transferability** -- A router trained solely on tool data generalizes to multi-agent collaboration with minimal adaptation.
- **Robustness & Scalability** -- Maintains exceptional robustness against noise and scales effectively to massive candidate spaces.

## Code

> **Code is coming within a few weeks.**
