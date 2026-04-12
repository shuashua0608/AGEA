# AGEA (Agentic Graph Extraction Attack)

Official implementation for [Query-Efficient Agentic Graph Extraction Attacks on GraphRAG Systems](https://arxiv.org/pdf/2601.14662) (ACL Main 2026), with unified runners for GraphRAG and LightRAG.

## Quick Start

### 1. Install dependencies

Set up GraphRAG and LightRAG cores first:
- GraphRAG: https://microsoft.github.io/graphrag/get_started/
- LightRAG: https://github.com/HKUDS/LightRAG

Example environment setup:

```bash
conda create -n agea python=3.10 -y
conda activate agea
python -m pip install graphrag openai python-dotenv networkx pandas pyyaml numpy
```

### 2. Prepare datasets

Dataset sources:
- Medical and Novel: [GraphRAG-Bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/tree/main/Datasets)
- Agriculture: [DIGIMON](https://github.com/JayLZhou/GraphRAG/tree/master?tab=readme-ov-file)

### 3. Required folder layout

GraphRAG run expects:
- `graphrag/<dataset>/settings.yaml`
- `graphrag/<dataset>/output/` (prebuilt GraphRAG index)

LightRAG build/run expects:
- `LightRAG/<dataset>/Corpus.json` (input corpus for builder)
- `LightRAG/graphs/<dataset>/` (generated LightRAG workspace)

### 4. Configure environment variables

Create a `.env` file at repo root with:

```bash
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=...

# query generation / filter deployments
QUERY_GENERATOR=...
GRAPH_FILTER_AGENT=...   # used by graphrag/run_agea.py
GRAPH_FILTER=...         # used by LightRAG/run_agea_lightrag.py

# LightRAG runner only
AZURE_OPENAI_DEPLOYMENT=...
AZURE_EMBEDDING_DEPLOYMENT=...
AZURE_EMBEDDING_API_VERSION=...
```

## Build LightRAG Graph Workspace

Run this once per dataset (or after changing `Corpus.json`):

```bash
python LightRAG/lightrag_azure_data.py --dataset medical
```

This creates `LightRAG/graphs/medical/`.

## Run AGEA on GraphRAG

Example:

```bash
python graphrag/run_agea.py --dataset medical --turns 3 --query-method local
```

Other query methods:

```bash
python graphrag/run_agea.py --dataset medical --turns 3 --query-method global
python graphrag/run_agea.py --dataset medical --turns 3 --query-method basic
```

Resume or ablation:

```bash
python graphrag/run_agea.py --dataset medical --turns 50 --resume
python graphrag/run_agea.py --dataset medical --turns 3 --disable-graph-filter
```

## Run AGEA on LightRAG

Example:

```bash
python LightRAG/run_agea_lightrag.py --dataset medical --turns 3 --query_method hybrid
```

Other modes:

```bash
python LightRAG/run_agea_lightrag.py --dataset medical --turns 3 --query_method local #global/naive/mix
```

Resume or ablation:

```bash
python LightRAG/run_agea_lightrag.py --dataset medical --turns 50 --resume
python LightRAG/run_agea_lightrag.py --dataset medical --turns 3 --disable-graph-filter
```

## Output Paths

GraphRAG outputs:

`graphrag/<dataset>/init_eps_..._<graph_filter|no_graph_filter>/<victim_model>/<query_method>/<run_name>/`

LightRAG outputs:

`LightRAG/graphs/<dataset>/init_eps_..._<graph_filter|no_graph_filter>/<victim_model>/<query_method>/<run_name>/`

Common artifacts:
- `extracted_graph.json` and `extracted_graph.graphml` (filtered graph)
- `query_history.json` (turn-by-turn records and cumulative metrics)
- `turn_logs/...` (retrieved contexts, LLM responses, graph-filter logs)
- `extraction_analysis.json` (final report)

## Main Scripts

- `graphrag/run_agea.py`: GraphRAG AGEA runner
- `LightRAG/run_agea_lightrag.py`: LightRAG AGEA runner
- `LightRAG/lightrag_azure_data.py`: LightRAG graph builder from `Corpus.json`
- `utils.py`: shared pathing/parsing/metrics helpers
- `agea_prompts.py`: prompt templates
- `graph_extractor_memory.py`: graph memory manager
- `graph_query_memory.py`: query memory manager
