# CrossGuard

**Multi-Agent Cross-Contract Vulnerability Detection for Ethereum DApps**

> Companion repository for the paper submitted to *ACM Transactions on Software Engineering and Methodology* (TOSEM).

---

## Overview

CrossGuard is a three-phase multi-agent framework that detects vulnerabilities arising from *interactions between smart contracts* in decentralised applications (DApps). Unlike existing detectors that analyse contracts in isolation, CrossGuard operates at the DApp level by constructing a typed interaction graph and orchestrating iterative cross-contract reasoning among per-contract LLM agents.

**Key results on DAppSCAN (682 DApps):**
- **83.93% F1**, exceeding the per-contract baseline by +7.57 F1 points (*p* = 0.008)
- **186 cross-contract vulnerabilities** detected that isolated analysis misses entirely
- Competitive on single-contract benchmarks: 89.05% F1 on ESC, 83.04% on SMS

## Architecture

```
Phase 1                    Phase 2                      Phase 3
Decomposition &            Cross-Contract               Vulnerability
Local Analysis             Reasoning Protocol           Synthesis
                                                        
DApp Source Files          Round 0: Broadcast           Aggregate
       |                         |                         |
Regex-Based Parser         Rounds 1...T (T=3)      Composite Score
       |                   LLM + Pattern Fallback        |
Interaction Graph               |                  Threshold Filter
G = (V, E)                 Convergence Check             |
6 typed edges                   |                  Deduplicate & Rank
       |                   Multi-Hop Detection           |
Agent Pool                      |                  Vulnerability
1 agent / contract         Cross-Contract           Report R
       |                   Findings
Agent Summaries
{S1, ..., Sn}

              LoRA Fine-Tuning (QLoRA)
              DeepSeek-Coder-6.7B
              rank 16, alpha=32, 4.19M params (0.06%)
```

## Repository Structure

```
crossguard/
├── config.py           # Configuration management (dataclasses + YAML)
├── decomposer.py       # DApp decomposition and interaction graph G=(V,E)
├── agents.py           # Per-contract LLM agents (local + cross analysis)
├── protocol.py         # Cross-contract reasoning protocol (Algorithm 1)
├── engine.py           # Orchestration pipeline, training, and evaluation
├── data.py             # Multi-dataset loading (ESC, SMS, DAppSCAN)
├── experiments.py      # Experiment runner for RQ1–RQ7
├── default.yaml        # Default configuration (matches paper Section 4.3)
├── requirements.txt    # Python dependencies
├── figures/            # Publication-quality figures (PDF + PNG + scripts)
└── README.md
```

### Module Descriptions

| Module | Section | Role |
|--------|---------|------|
| `config.py` | §4.3 | Typed configuration via dataclasses; loads/merges YAML files; all defaults match the paper |
| `decomposer.py` | §3.2 | Regex-based parser extracting contracts, functions, state variables; builds interaction graph with 6 edge types (`EXTERNAL_CALL`, `INHERITANCE`, `INTERFACE_DEP`, `STATE_DEPENDENCY`, `EVENT_DEPENDENCY`, `PROXY_DELEGATE`) |
| `agents.py` | §3.2 | `ContractAgent` class: one agent per contract, LLM-driven local analysis with regex pattern fallback, cross-contract reasoning; shared DeepSeek-Coder-6.7B backbone with QLoRA |
| `protocol.py` | §3.3 | Algorithm 1 implementation: Round 0 initial exchange, Rounds 1–*T* iterative reasoning, convergence tracking (cosine similarity), multi-hop attack path detection (DFS), 8 vulnerability pattern families, Phase 3 scoring |
| `engine.py` | §3–4 | `CrossGuardPipeline` (three-phase orchestration), `AgentTrainer` (LoRA fine-tuning with prompt-masked loss), `BaselineRunner` (ablation baselines), `CrossGuardEvaluator` (end-to-end evaluation) |
| `data.py` | §4.1 | Dataset loaders for ESC (9,742 contracts), SMS (514,880 functions), and DAppSCAN (682 DApps); stratified splitting; DApp wrapping for single-contract datasets |
| `experiments.py` | §5 | Orchestrates RQ1–RQ7: effectiveness, cross-contract value, ablation, scalability, convergence, case studies, efficiency |

## Requirements

- Python 3.10 or 3.11
- CUDA 11.8 or 12.1
- A single GPU with ≥24 GB VRAM (tested on NVIDIA A100 40GB)

### Installation

```bash
git clone this GitHub link
cd crossguard
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.1, <2.5 | Core ML framework |
| `transformers` | ≥4.36, <4.46 | DeepSeek-Coder-6.7B backbone |
| `peft` | ≥0.7, <0.14 | LoRA / QLoRA adapters |
| `bitsandbytes` | ≥0.41, <0.45 | 4-bit quantisation |
| `accelerate` | ≥0.25, <0.35 | Device mapping |
| `scikit-learn` | ≥1.3, <1.6 | Metrics, stratified splitting |
| `pyyaml` | ≥6.0 | Configuration |
| `tqdm` | ≥4.66 | Progress bars |
| `matplotlib` | ≥3.7 | Figure generation (optional) |

## Datasets

Place datasets under `data/` following this structure:

```
data/
├── dataset1_esc/
│   └── raw/              # 9,742 Solidity contracts
├── dataset2_sms/
│   └── raw/              # 514,880 function-level samples
└── dataset3_dappscan/
    ├── source/           # 682 DApps (multi-contract source)
    └── bytecode/         # Compiled bytecode (optional)
```

| Dataset | Granularity | Size | Split | Reference |
|---------|------------|------|-------|-----------|
| ESC | Contract | 9,742 | 70/15/15 stratified | Zhuang et al., IJCAI 2020 |
| SMS | Function | 514,880 | 80/20 × 5 runs | Qian et al., WWW 2023 |
| DAppSCAN | DApp | 682 (multi-contract) | 70/15/15 stratified | DAppSCAN public audit reports |

## Usage

### Full Experiment Pipeline (RQ1–RQ7)

```bash
python experiments.py --config default.yaml
```

### Individual Phases

```python
from config import load_config
from data import build_dataloaders
from engine import CrossGuardPipeline

cfg = load_config("default.yaml")
train_dl, val_dl, test_dl = build_dataloaders(cfg)

pipeline = CrossGuardPipeline(cfg)
report = pipeline.run(test_dl)
```

### LoRA Fine-Tuning Only

```python
from engine import AgentTrainer

trainer = AgentTrainer(cfg)
trainer.train(train_dl, val_dl)
# Best checkpoint saved to results/checkpoints/
```

### Configuration

All hyperparameters are controlled via `default.yaml`. Key settings:

```yaml
agent:
  llm_model: "deepseek-ai/deepseek-coder-6.7b-instruct"
  lora_rank: 16
  lora_alpha: 32

protocol:
  num_reasoning_rounds: 3       # T = 3
  convergence_threshold: 0.05   # theta
  vulnerability_score_threshold: 0.5

training:
  lora_finetune_lr: 2.0e-5
  lora_finetune_epochs: 10
```

Override any setting via CLI:

```bash
python experiments.py --config default.yaml \
    --override agent.lora_rank=32 \
    --override protocol.num_reasoning_rounds=5
```

## Research Questions

| RQ | Question | Key Finding |
|----|----------|-------------|
| RQ1 | Overall effectiveness? | 83.93% F1 on DAppSCAN, 89.05% on ESC, 83.04% on SMS |
| RQ2 | Value of cross-contract reasoning? | +4.65 F1 over local-only; 186 cross-contract vulns detected |
| RQ3 | Component contributions? | Interaction graph is most impactful (−7.57 F1 when removed) |
| RQ4 | Scalability with DApp size? | F1 stable across 2–21 contract DApps |
| RQ5 | How many reasoning rounds? | T=3 is optimal; beyond T=3, over-reasoning degrades precision |
| RQ6 | Qualitative case study? | UniLend_Finance: 3 cross-contract vulns found across 6 contracts |
| RQ7 | Runtime efficiency? | Phase 2 dominates; total time scales linearly with edge count |

## Figures

Publication-quality figures are in `figures/`. Each has a Python generation script, a 600 dpi PNG, and a camera-ready PDF (TrueType embedding, Liberation Serif font):

| Figure | File | Description |
|--------|------|-------------|
| Architecture | `architecture.pdf` | End-to-end three-phase pipeline |
| Motivating example | `motivating_example.drawio` | Cross-reentrancy between two contracts |
| RQ2 | `rq2_cross_value.pdf` | Grouped bar chart: full vs local-only vs cross-only |
| RQ3 | `rq3_ablation.pdf` | Horizontal bar chart of F1 drops per ablation |
| RQ5 | `rq5_convergence.pdf` | Two-panel: F1 vs T + per-DApp convergence curves |
| RQ6 | `rq6_casestudy.pdf` | UniLend_Finance interaction graph with vulnerability callouts |

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | DeepSeek-Coder-6.7B-Instruct |
| Quantisation | QLoRA (4-bit NF4) |
| LoRA rank / alpha | 16 / 32 |
| Trainable parameters | 4.19M (0.06% of 6.7B) |
| Learning rate | 2 × 10⁻⁵ (AdamW) |
| Effective batch size | 32 (4 × 8 gradient accumulation) |
| Loss | Prompt-masked cross-entropy |
| Best checkpoint | Epoch 8 of 10 |
| Training time | 4.2 hours on 1× NVIDIA A100 40GB |

## Reproducibility

All experiments use seed 42 with deterministic operations enabled. Statistical significance is assessed via permutation testing (10,000 permutations) and bootstrap confidence intervals (1,000 samples), as described in Section 4.4 of the paper.

```yaml
reproducibility:
  seed: 42
  deterministic: true
  num_bootstrap_samples: 1000
  num_permutation_tests: 10000
```

## License

This repository is released for academic research purposes. See `LICENSE` for details.
