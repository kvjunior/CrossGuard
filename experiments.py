"""
Experiment orchestration for CrossGuard.

Seven research questions (SS5):
  RQ1  Effectiveness     three datasets + ablation baselines
  RQ2  Cross-Contract    full vs local-only vs cross-only
  RQ3  Ablation          seven-component ablation
  RQ4  Scalability       performance vs DApp size
  RQ5  Protocol Rounds   convergence analysis
  RQ6  Case Studies      structured qualitative analysis
  RQ7  Efficiency        runtime breakdown by phase

Author : [Anonymous for double-blind review]
Target : ACM Transactions on Software Engineering and Methodology
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

from .config import Config, get_default_config, create_ablation_configs
from .agents import LLMModule
from .data import create_dataloaders
from .engine import (CrossGuardPipeline, CrossGuardEvaluator,
                     AgentTrainer, ReportedBaselines)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def setup_logging(config):
    """Configure root logger with console + file handler."""
    level = getattr(logging,
                    config.logging_cfg.log_level.upper(),
                    logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    log_dir = config.paths.resolve("logs_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "crossguard.log")
    fh.setLevel(level)
    logging.getLogger().addHandler(fh)


def get_system_info() -> Dict[str, Any]:
    """Collect hardware/software environment for reproducibility."""
    info: Dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = [torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())]
        info["gpu_memory_mb"] = [
            torch.cuda.get_device_properties(i).total_mem // (1024**2)
            for i in range(torch.cuda.device_count())]
    return info


def _save(data: Any, path: Path):
    """Save JSON with Path/numpy serialisation support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug(f"Saved: {path}")


def _resolve(config, dir_name: str) -> Path:
    """Resolve a PathConfig directory to an absolute Path."""
    return config.paths.resolve(dir_name)


# ═══════════════════════════════════════════════════════════════════════
# Core evaluation helper
# ═══════════════════════════════════════════════════════════════════════

def _eval_dataset(config, dataset: str,
                  llm: Optional[LLMModule] = None,
                  tag: str = "",
                  train_lora: bool = False,
                  ) -> tuple:
    """Load data, optionally fine-tune, evaluate, and save results.

    Returns (results_dict, evaluator, loaders).
    """
    loaders = create_dataloaders(config, dataset)
    pipeline = CrossGuardPipeline(config, llm)
    evaluator = CrossGuardEvaluator(config, pipeline, llm)

    # ── optional LoRA fine-tuning ──────────────────────────────────
    train_history: Dict[str, Any] = {}
    if (train_lora and llm is not None
            and not config.agent.use_fallback_mlp):
        trainer = AgentTrainer(config, llm)
        train_history = trainer.train(
            loaders["train"], loaders.get("val"))

    # ── evaluate on test set ───────────────────────────────────────
    results = evaluator.evaluate(loaders["test"])
    results["dataset"] = dataset
    results["system_info"] = get_system_info()
    results["train_history"] = train_history

    # ── attach reported baseline numbers ───────────────────────────
    results["reported_baselines"] = \
        ReportedBaselines.get_for_dataset(dataset)
    results["comparison_table"] = \
        ReportedBaselines.generate_comparison_table(
            results, dataset)

    # ── save ───────────────────────────────────────────────────────
    fname = f"{tag}_{dataset}.json" if tag else \
        f"{dataset}_results.json"
    _save(results, _resolve(config, "metrics_dir") / fname)
    return results, evaluator, loaders


# ═══════════════════════════════════════════════════════════════════════
# RQ1: Effectiveness across three datasets
# ═══════════════════════════════════════════════════════════════════════

class CrossDatasetExperiment:
    """RQ1 (SS5.1): evaluate on ESC, SMS, DAppSCAN with baselines.

    For SMS, the paper prescribes "80/20 train/test split with five
    independent runs, results averaged across runs" (SS4.1).
    """

    def __init__(self, config, llm=None):
        self.config = config
        self.llm = llm

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("RQ1: Effectiveness + Baselines")
        logger.info("=" * 60)

        results: Dict[str, Any] = {}
        for ds in self.config.data.active_datasets:
            logger.info(f"\n-- Dataset: {ds} --")

            if ds == "sms":
                r = self._run_sms_multi_run()
                results[ds] = r
            else:
                r, evaluator, loaders = _eval_dataset(
                    self.config, ds, self.llm,
                    "rq1", train_lora=True)
                results[ds] = r
                if self.config.experiment.run_baselines:
                    try:
                        bl = evaluator.evaluate_baselines(
                            loaders["test"])
                        results[f"{ds}_baselines"] = bl
                    except Exception as exc:
                        logger.warning(
                            f"Baselines failed for {ds}: {exc}")

        _save(results,
              _resolve(self.config, "metrics_dir")
              / "rq1_effectiveness.json")
        return results

    def _run_sms_multi_run(self) -> Dict[str, Any]:
        """Run SMS evaluation N times with different seeds and average.

        Paper (SS4.1): "80/20 train/test split with five independent
        runs, results averaged across runs."
        """
        n_runs = self.config.data.sms_num_runs
        all_metrics: Dict[str, List[float]] = defaultdict(list)

        for run_idx in range(n_runs):
            logger.info(f"  SMS run {run_idx + 1}/{n_runs}")
            cfg_run = Config(self.config.to_dict())
            cfg_run.reproducibility.seed = (
                self.config.reproducibility.seed + run_idx)

            r, _, _ = _eval_dataset(
                cfg_run, "sms", self.llm,
                f"rq1_sms_run{run_idx}", train_lora=True)

            m = r.get("metrics", {})
            for k in ("accuracy", "precision", "recall",
                      "f1_score", "auroc"):
                all_metrics[k].append(m.get(k, 0.0))

        # Average across runs
        averaged: Dict[str, Any] = {
            "metrics": {},
            "n_runs": n_runs,
            "per_run_metrics": dict(all_metrics),
        }
        for k, vals in all_metrics.items():
            averaged["metrics"][k] = float(np.mean(vals))
            averaged["metrics"][f"{k}_std"] = float(np.std(vals))

        logger.info(
            f"  SMS averaged F1: "
            f"{averaged['metrics'].get('f1_score', 0):.4f} "
            f"(+/- {averaged['metrics'].get('f1_score_std', 0):.4f})")
        return averaged


# ═══════════════════════════════════════════════════════════════════════
# RQ2: Value of cross-contract reasoning
# ═══════════════════════════════════════════════════════════════════════

class CrossContractValueExperiment:
    """RQ2 (SS5.2): full vs local-only vs cross-only on DAppSCAN.

    Three configurations:
      full       complete CrossGuard pipeline
      local-only Phase 2 disabled (T=0)
      cross-only Phase 1 local analysis disabled (token budget = 0)
    """

    def __init__(self, config, llm=None):
        self.config = config
        self.llm = llm

    def run(self, dataset: str = "dappscan") -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("RQ2: Cross-Contract Value")
        logger.info("=" * 60)

        # ── full model (train once, reuse) ─────────────────────────
        r_full, _, _ = _eval_dataset(
            self.config, dataset, self.llm,
            "rq2_full", train_lora=True)

        # ── local-only (T=0) ───────────────────────────────────────
        cfg_local = Config(self.config.to_dict())
        cfg_local.protocol.num_reasoning_rounds = 0
        r_local, _, _ = _eval_dataset(
            cfg_local, dataset, self.llm, "rq2_local")

        # ── cross-only (no Phase 1 local analysis) ─────────────────
        cfg_cross = Config(self.config.to_dict())
        cfg_cross.agent.local_analysis_max_tokens = 0
        r_cross, _, _ = _eval_dataset(
            cfg_cross, dataset, self.llm, "rq2_cross")

        fm = r_full.get("metrics", {})
        lm = r_local.get("metrics", {})
        cm = r_cross.get("metrics", {})

        results = {
            "full": fm,
            "local_only": lm,
            "cross_only": cm,
            "delta_f1_full_vs_local": (
                fm.get("f1_score", 0) - lm.get("f1_score", 0)),
            "delta_f1_full_vs_cross": (
                fm.get("f1_score", 0) - cm.get("f1_score", 0)),
            "cross_contract_recall_full": fm.get(
                "cross_contract_recall", 0),
            "cross_contract_recall_local": lm.get(
                "cross_contract_recall", 0),
            "cross_contract_recall_cross": cm.get(
                "cross_contract_recall", 0),
        }

        logger.info(
            f"  F1: full={fm.get('f1_score', 0):.4f}, "
            f"local={lm.get('f1_score', 0):.4f}, "
            f"cross={cm.get('f1_score', 0):.4f}")
        logger.info(
            f"  Delta F1 (full - local): "
            f"+{results['delta_f1_full_vs_local']:.4f}")

        _save(results,
              _resolve(self.config, "metrics_dir")
              / "rq2_cross_value.json")
        return results


# ═══════════════════════════════════════════════════════════════════════
# RQ3: Ablation study
# ═══════════════════════════════════════════════════════════════════════

class AblationExperiment:
    """RQ3 (SS5.3): seven-component ablation on DAppSCAN.

    Constructs seven ablation variants via create_ablation_configs(),
    evaluates each on the same test set, and computes F1 deltas.
    """

    def __init__(self, config, llm=None):
        self.config = config
        self.llm = llm

    def run(self, dataset: str = "dappscan") -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("RQ3: Ablation Study")
        logger.info("=" * 60)

        # ── full model ─────────────────────────────────────────────
        r_full, _, _ = _eval_dataset(
            self.config, dataset, self.llm,
            "abl_full", train_lora=True)
        fm = r_full.get("metrics", {})

        # ── ablation variants ──────────────────────────────────────
        variants = create_ablation_configs(self.config)
        results: Dict[str, Any] = {"full_model": fm}

        for name, vcfg in variants.items():
            logger.info(f"\n-- Ablation: {name} --")
            try:
                vr, _, _ = _eval_dataset(
                    vcfg, dataset, self.llm, f"abl_{name}")
                results[name] = vr.get("metrics", {})
            except Exception as exc:
                logger.warning(f"Ablation {name} failed: {exc}")
                results[name] = {"error": str(exc)}

        # ── compute deltas ─────────────────────────────────────────
        deltas: Dict[str, Dict[str, float]] = {}
        for name, vm in results.items():
            if name in ("full_model", "deltas"):
                continue
            if not isinstance(vm, dict) or "f1_score" not in vm:
                continue
            deltas[name] = {
                k: fm.get(k, 0) - vm.get(k, 0)
                for k in ("accuracy", "precision",
                          "recall", "f1_score")}

        results["deltas"] = deltas

        # Log sorted by impact
        for name, d in sorted(deltas.items(),
                              key=lambda x: x[1].get("f1_score", 0),
                              reverse=True):
            logger.info(
                f"  {name}: delta_F1 = "
                f"{d.get('f1_score', 0) * 100:+.2f} pp")

        _save(results,
              _resolve(self.config, "metrics_dir")
              / "rq3_ablation.json")
        return results


# ═══════════════════════════════════════════════════════════════════════
# RQ4: Scalability
# ═══════════════════════════════════════════════════════════════════════

class ScalabilityExperiment:
    """RQ4 (SS5.4): detection quality and runtime vs DApp size.

    Partitions the DAppSCAN test set into four buckets by the number
    of contracts and evaluates each independently.
    """

    BUCKET_LABELS = ["1", "2-5", "6-20", "21+"]

    def __init__(self, config, llm=None):
        self.config = config
        self.llm = llm

    def run(self, dataset: str = "dappscan") -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("RQ4: Scalability")
        logger.info("=" * 60)

        loaders = create_dataloaders(self.config, dataset)
        pipeline = CrossGuardPipeline(self.config, self.llm)

        buckets: Dict[str, Dict[str, list]] = {
            bk: {"times": [], "preds": [], "targets": []}
            for bk in self.BUCKET_LABELS}

        for batch in loaders["test"]:
            for s in batch:
                r = pipeline.analyse_dapp(
                    s["files"], s["dapp_id"])
                n = r["n_contracts"]
                if n == 1:
                    bk = "1"
                elif n <= 5:
                    bk = "2-5"
                elif n <= 20:
                    bk = "6-20"
                else:
                    bk = "21+"
                buckets[bk]["times"].append(r["time_s"])
                buckets[bk]["preds"].append(r["prediction"])
                buckets[bk]["targets"].append(s["label"])

        # Compute per-bucket metrics (Table IV)
        results: Dict[str, Any] = {}
        for bk in self.BUCKET_LABELS:
            d = buckets[bk]
            if not d["targets"]:
                continue
            y = np.array(d["targets"])
            yh = np.array(d["preds"])
            results[bk] = {
                "n": len(y),
                "accuracy": float(accuracy_score(y, yh)),
                "precision": float(precision_score(
                    y, yh, zero_division=0)),
                "recall": float(recall_score(
                    y, yh, zero_division=0)),
                "f1": float(f1_score(
                    y, yh, zero_division=0)),
                "avg_time_s": float(np.mean(d["times"])),
            }
            logger.info(
                f"  {bk}: n={results[bk]['n']}, "
                f"F1={results[bk]['f1']:.4f}, "
                f"time={results[bk]['avg_time_s']:.2f}s")

        _save(results,
              _resolve(self.config, "metrics_dir")
              / "rq4_scalability.json")
        return results


# ═══════════════════════════════════════════════════════════════════════
# RQ5: Protocol rounds + convergence
# ═══════════════════════════════════════════════════════════════════════

class ProtocolRoundsExperiment:
    """RQ5 (SS5.5): effect of reasoning rounds on detection quality.

    Trains once with T=3, then evaluates with T in {0,1,2,3,5,10}
    using the same trained model.  Also collects per-DApp convergence
    curves for the default T=3 setting.
    """

    ROUND_VALUES = [0, 1, 2, 3, 5, 10]

    def __init__(self, config, llm=None):
        self.config = config
        self.llm = llm

    def run(self, dataset: str = "dappscan") -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("RQ5: Protocol Rounds + Convergence")
        logger.info("=" * 60)

        # ── train once with default T ──────────────────────────────
        if (self.llm is not None
                and not self.config.agent.use_fallback_mlp):
            loaders_train = create_dataloaders(
                self.config, dataset)
            trainer = AgentTrainer(self.config, self.llm)
            trainer.train(
                loaders_train["train"],
                loaders_train.get("val"))

        # ── evaluate across T values ───────────────────────────────
        results: Dict[str, Any] = {}
        for T in self.ROUND_VALUES:
            cfg = Config(self.config.to_dict())
            cfg.protocol.num_reasoning_rounds = T
            logger.info(f"\n-- T={T} --")
            r, _, _ = _eval_dataset(
                cfg, dataset, self.llm, f"rq5_T{T}")
            m = r.get("metrics", {})
            results[f"T={T}"] = {
                "rounds": T,
                "f1": m.get("f1_score", 0),
                "cross_findings": m.get("total_cross", 0),
                "cc_recall": m.get("cross_contract_recall", 0),
                "time_s": m.get("avg_time_s", 0),
            }

        # ── collect convergence curves at T=3 ─────────────────────
        cfg3 = Config(self.config.to_dict())
        cfg3.protocol.num_reasoning_rounds = 3
        cfg3.protocol.track_message_similarity = True
        loaders = create_dataloaders(cfg3, dataset)
        pipeline = CrossGuardPipeline(cfg3, self.llm)

        convergence_curves: List[Dict[str, Any]] = []
        n_collected = 0
        for batch in loaders["test"]:
            for s in batch:
                if s["num_contracts"] < 2:
                    continue  # only multi-contract DApps
                r = pipeline.analyse_dapp(
                    s["files"], s["dapp_id"])
                conv = r.get("convergence", {})
                if conv.get("curve"):
                    convergence_curves.append({
                        "dapp": s["dapp_id"],
                        "curve": conv["curve"],
                        "n_rounds": conv.get(
                            "n_rounds_used", 0),
                    })
                    n_collected += 1
                if n_collected >= 5:
                    break
            if n_collected >= 5:
                break

        results["convergence_curves"] = convergence_curves

        _save(results,
              _resolve(self.config, "metrics_dir")
              / "rq5_rounds.json")
        return results


# ═══════════════════════════════════════════════════════════════════════
# RQ6: Case studies
# ═══════════════════════════════════════════════════════════════════════

class CaseStudyExperiment:
    """RQ6 (SS5.6): qualitative case studies from DAppSCAN test set."""

    def __init__(self, config, llm=None):
        self.config = config
        self.llm = llm

    def run(self, dataset: str = "dappscan") -> List[Dict]:
        logger.info("=" * 60)
        logger.info("RQ6: Case Studies")
        logger.info("=" * 60)

        loaders = create_dataloaders(self.config, dataset)
        pipeline = CrossGuardPipeline(self.config, self.llm)
        evaluator = CrossGuardEvaluator(
            self.config, pipeline, self.llm)

        cases = evaluator.generate_case_studies(
            loaders["test"], n=10)

        # ── generate LaTeX for paper ───────────────────────────────
        latex_parts = [r"\subsection{Case Studies}"]
        for cs in cases[:3]:
            latex_parts.append(cs.to_latex())
            latex_parts.append("")

        tables_dir = _resolve(self.config, "tables_dir")
        latex_path = tables_dir / "case_studies.tex"
        with open(latex_path, "w") as f:
            f.write("\n".join(latex_parts))

        # ── serialise for JSON output ──────────────────────────────
        serialised: List[Dict[str, Any]] = []
        for cs in cases:
            serialised.append({
                "dapp_id": cs.dapp_id,
                "num_contracts": cs.num_contracts,
                "graph_summary": cs.graph_summary,
                "n_cross_findings": len(
                    cs.cross_contract_findings),
                "cross_findings": cs.cross_contract_findings[:5],
                "what_would_be_missed": (
                    cs.what_single_contract_would_miss),
                "convergence_rounds": cs.convergence_rounds,
                "ground_truth": cs.ground_truth[:5],
            })

        _save(serialised,
              _resolve(self.config, "metrics_dir")
              / "rq6_cases.json")
        logger.info(
            f"  {len(cases)} case studies, LaTeX -> {latex_path}")
        return serialised


# ═══════════════════════════════════════════════════════════════════════
# RQ7: Efficiency
# ═══════════════════════════════════════════════════════════════════════

class EfficiencyExperiment:
    """RQ7 (SS5.7): runtime and memory cost analysis.

    Paper (Table VII): reports mean, median, P95 runtime and GPU
    memory for each method on the DAppSCAN test set.
    """

    def __init__(self, config, llm=None):
        self.config = config
        self.llm = llm

    def run(self, dataset: str = "dappscan") -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("RQ7: Efficiency")
        logger.info("=" * 60)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        loaders = create_dataloaders(self.config, dataset)
        pipeline = CrossGuardPipeline(self.config, self.llm)
        times: List[float] = []

        for batch in loaders["test"]:
            for s in batch:
                t0 = time.time()
                pipeline.analyse_dapp(
                    s["files"], s["dapp_id"])
                times.append(time.time() - t0)

        gpu_mb = 0.0
        if torch.cuda.is_available():
            gpu_mb = (torch.cuda.max_memory_allocated()
                      / (1024 ** 2))

        results: Dict[str, Any] = {
            "n_dapps": len(times),
            "total_s": sum(times),
            "mean_s": float(np.mean(times)) if times else 0,
            "median_s": float(np.median(times)) if times else 0,
            "p95_s": (float(np.percentile(times, 95))
                      if times else 0),
            "gpu_peak_mb": gpu_mb,
        }

        logger.info(
            f"  n={results['n_dapps']}, "
            f"mean={results['mean_s']:.2f}s, "
            f"median={results['median_s']:.2f}s, "
            f"P95={results['p95_s']:.2f}s, "
            f"GPU={results['gpu_peak_mb']:.0f}MB")

        _save(results,
              _resolve(self.config, "metrics_dir")
              / "rq7_efficiency.json")
        return results


# ═══════════════════════════════════════════════════════════════════════
# Figure generation
# ═══════════════════════════════════════════════════════════════════════

def generate_figures(config):
    """Generate PDF figures for the paper from saved experiment data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable — skipping figures")
        return

    fig_dir = _resolve(config, "figures_dir")
    met_dir = _resolve(config, "metrics_dir")

    # ── RQ2: cross-contract value (Figure 5) ──────────────────────
    fp = met_dir / "rq2_cross_value.json"
    if fp.exists():
        with open(fp) as f:
            data = json.load(f)
        modes = ["full", "local_only", "cross_only"]
        labels = ["CrossGuard\n(Full)",
                   "Local Only\n(No Phase 2)",
                   "Cross Only\n(No Phase 1)"]
        metrics = ["precision", "recall", "f1_score"]
        x = np.arange(len(modes))
        w = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, m_name in enumerate(metrics):
            vals = [data.get(mode, {}).get(m_name, 0)
                    for mode in modes]
            ax.bar(x + (i - 1) * w, vals, w,
                   label=m_name.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("RQ2: Value of Cross-Contract Reasoning")
        plt.tight_layout()
        plt.savefig(fig_dir / "rq2_cross_value.pdf", dpi=300)
        plt.close()

    # ── RQ3: ablation (Figure 6) ──────────────────────────────────
    fp = met_dir / "rq3_ablation.json"
    if fp.exists():
        with open(fp) as f:
            data = json.load(f)
        deltas = data.get("deltas", {})
        if deltas:
            # Sort by impact (largest drop first)
            sorted_items = sorted(
                deltas.items(),
                key=lambda x: x[1].get("f1_score", 0),
                reverse=True)
            names = [n for n, _ in sorted_items]
            vals = [d.get("f1_score", 0) * 100
                    for _, d in sorted_items]
            x = np.arange(len(names))
            fig, ax = plt.subplots(figsize=(12, 5))
            colors = ["#F44336" if v > 0 else "#4CAF50"
                      for v in vals]
            ax.barh(x, vals, color=colors)
            ax.set_yticks(x)
            ax.set_yticklabels(
                [n.replace("no_", "w/o ").replace("_", " ")
                 for n in names],
                fontsize=9)
            ax.set_xlabel("F1 Drop (percentage points)")
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title("RQ3: Ablation — F1 Drop per Component")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / "rq3_ablation.pdf", dpi=300)
            plt.close()

    # ── RQ5: protocol rounds + convergence (Figure 7) ─────────────
    fp = met_dir / "rq5_rounds.json"
    if fp.exists():
        with open(fp) as f:
            data = json.load(f)
        rounds_data = {k: v for k, v in data.items()
                       if k.startswith("T=")}
        if rounds_data:
            rs = sorted(
                rounds_data.keys(),
                key=lambda k: rounds_data[k].get("rounds", 0))
            T_vals = [rounds_data[k]["rounds"] for k in rs]
            f1s = [rounds_data[k].get("f1", 0) for k in rs]

            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(14, 5))
            ax1.plot(T_vals, f1s, "o-", lw=2, ms=8,
                     color="#F44336")
            ax1.set_xlabel("Reasoning Rounds (T)")
            ax1.set_ylabel("F1 Score")
            ax1.set_title("RQ5: F1 vs Rounds")
            ax1.grid(alpha=0.3)

            # Convergence curves
            curves = data.get("convergence_curves", [])
            for i, cc in enumerate(curves[:5]):
                curve = cc.get("curve", [])
                if curve:
                    lbl = cc.get("dapp", f"DApp {i}")[:15]
                    ax2.plot(range(1, len(curve) + 1), curve,
                             "o-", ms=5, label=lbl, alpha=0.7)
            ax2.set_xlabel("Round")
            ax2.set_ylabel("Message Similarity")
            ax2.set_title("RQ5: Convergence Curves")
            ax2.legend(fontsize=7)
            ax2.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / "rq5_rounds.pdf", dpi=300)
            plt.close()

    # ── RQ4: scalability (Figure 8) ───────────────────────────────
    fp = met_dir / "rq4_scalability.json"
    if fp.exists():
        with open(fp) as f:
            data = json.load(f)
        bks = [b for b in ScalabilityExperiment.BUCKET_LABELS
               if b in data]
        if bks:
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(12, 5))
            ax1.bar(bks,
                    [data[b].get("f1", 0) for b in bks],
                    color="#4CAF50")
            ax1.set_ylabel("F1")
            ax1.set_xlabel("Contracts per DApp")
            ax1.set_title("Detection Quality vs Size")
            ax1.grid(axis="y", alpha=0.3)

            ax2.bar(bks,
                    [data[b].get("avg_time_s", 0) for b in bks],
                    color="#2196F3")
            ax2.set_ylabel("Time (s)")
            ax2.set_xlabel("Contracts per DApp")
            ax2.set_title("Runtime vs Size")
            ax2.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            plt.savefig(fig_dir / "rq4_scalability.pdf", dpi=300)
            plt.close()

    logger.info(f"Figures saved to {fig_dir}")


# ═══════════════════════════════════════════════════════════════════════
# LaTeX table generation
# ═══════════════════════════════════════════════════════════════════════

def generate_latex_tables(config):
    """Generate LaTeX tables for the paper from saved experiment data."""
    met_dir = _resolve(config, "metrics_dir")
    tab_dir = _resolve(config, "tables_dir")

    # ── Table I: Main results with reported baselines ──────────────
    fp = met_dir / "rq1_effectiveness.json"
    if fp.exists():
        with open(fp) as f:
            data = json.load(f)
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{RQ1: CrossGuard effectiveness.  "
            r"Numbers marked $\dagger$ are cited from original "
            r"publications; all other rows are our own "
            r"measurements.}",
            r"\label{tab:main}",
            r"\small",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\renewcommand{\arraystretch}{1.12}",
            r"\begin{tabular}{@{}ll ccccc cc l@{}}",
            r"\toprule",
            r"\textbf{Dataset} & \textbf{Method} & "
            r"\textbf{Acc} & \textbf{Pre} & \textbf{Rec} "
            r"& \textbf{F1} & \textbf{AUROC} "
            r"& \textbf{Local} & \textbf{Cross} "
            r"& \textbf{Src} \\",
            r"\midrule",
        ]

        for ds in config.data.active_datasets:
            if ds not in data:
                continue
            ds_data = data[ds]
            comp = ds_data.get("comparison_table", [])

            for row in comp:
                method = row["method"]
                if row.get("is_ours"):
                    m = ds_data.get("metrics", {})
                    lines.append(
                        f"  {ds.upper()} & "
                        f"\\textbf{{{method}}} & "
                        f"\\textbf{{{row['accuracy']}}} & "
                        f"\\textbf{{{row['precision']}}} & "
                        f"\\textbf{{{row['recall']}}} & "
                        f"\\textbf{{{row['f1']}}} & "
                        f"{m.get('auroc', 0):.3f} & "
                        f"{m.get('total_local', 0)} & "
                        f"{m.get('total_cross', 0)} & "
                        f"ours \\\\")
                else:
                    cite = row.get("cite_key", "")
                    cite_str = (f"\\cite{{{cite}}}"
                                if cite else "cited")
                    lines.append(
                        f"  {ds.upper()} & "
                        f"{method}$^\\dagger$ & "
                        f"{row['accuracy']} & "
                        f"{row['precision']} & "
                        f"{row['recall']} & "
                        f"{row['f1']} & "
                        f"--- & --- & --- & "
                        f"{cite_str} \\\\")

            # Our ablation baselines
            bk = f"{ds}_baselines"
            if bk in data and isinstance(data[bk], dict):
                for bl_name, bl_m in data[bk].items():
                    if not isinstance(bl_m, dict):
                        continue
                    lines.append(
                        f"  {ds.upper()} & "
                        f"{bl_name} & "
                        f"{bl_m.get('accuracy', 0) * 100:.2f} & "
                        f"{bl_m.get('precision', 0) * 100:.2f} & "
                        f"{bl_m.get('recall', 0) * 100:.2f} & "
                        f"{bl_m.get('f1_score', 0) * 100:.2f} & "
                        f"--- & --- & --- & ours \\\\")
            lines.append(r"\midrule")

        if lines[-1] == r"\midrule":
            lines.pop()
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ])
        with open(tab_dir / "table_main.tex", "w") as f:
            f.write("\n".join(lines))

    # ── Table II: Ablation ─────────────────────────────────────────
    fp = met_dir / "rq3_ablation.json"
    if fp.exists():
        with open(fp) as f:
            data = json.load(f)
        deltas = data.get("deltas", {})
        fm = data.get("full_model", {})

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{RQ3: Ablation study on DAppSCAN.  "
            r"$\Delta$F1 reports the change in F1 relative "
            r"to the full model.}",
            r"\label{tab:ablation}",
            r"\small",
            r"\begin{tabular}{@{}l cccc r@{}}",
            r"\toprule",
            r"\textbf{Variant} & \textbf{Acc} & "
            r"\textbf{Pre} & \textbf{Rec} & \textbf{F1} "
            r"& \textbf{$\Delta$F1} \\",
            r"\midrule",
            f"  \\textbf{{CrossGuard (full)}} & "
            f"{fm.get('accuracy', 0):.3f} & "
            f"{fm.get('precision', 0):.3f} & "
            f"{fm.get('recall', 0):.3f} & "
            f"\\textbf{{{fm.get('f1_score', 0):.3f}}} & "
            f"--- \\\\",
            r"\midrule",
        ]

        # Sort by impact magnitude
        sorted_d = sorted(
            deltas.items(),
            key=lambda x: abs(x[1].get("f1_score", 0)),
            reverse=True)
        for name, dm in sorted_d:
            vm = data.get(name, {})
            if not isinstance(vm, dict) or "f1_score" not in vm:
                continue
            delta = dm.get("f1_score", 0)
            sign = "+" if delta >= 0 else ""
            label = (name.replace("no_", "w/o ")
                     .replace("_", " "))
            lines.append(
                f"  {label} & "
                f"{vm.get('accuracy', 0):.3f} & "
                f"{vm.get('precision', 0):.3f} & "
                f"{vm.get('recall', 0):.3f} & "
                f"{vm.get('f1_score', 0):.3f} & "
                f"${sign}{delta * 100:.1f}$ \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        with open(tab_dir / "table_ablation.tex", "w") as f:
            f.write("\n".join(lines))

    logger.info(f"Tables saved to {tab_dir}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="CrossGuard Experiments (ACM TOSEM)")
    p.add_argument(
        "--mode",
        choices=["eval", "rq1", "rq2", "rq3", "rq4",
                 "rq5", "rq6", "rq7",
                 "figures", "tables", "all"],
        default="eval",
        help="Which experiment to run")
    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file")
    p.add_argument(
        "--dataset", default=None,
        help="Override active dataset (esc, sms, dappscan)")
    p.add_argument(
        "--gpu", default=None,
        help="GPU device ID(s)")
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed")
    return p.parse_args()


def main():
    args = parse_args()

    # ── load config ────────────────────────────────────────────────
    if os.path.exists(args.config):
        config = Config.load(args.config)
    else:
        logger.info(
            f"Config file {args.config} not found, "
            f"using defaults")
        config = get_default_config()

    # ── apply CLI overrides ────────────────────────────────────────
    if args.seed is not None:
        config.reproducibility.seed = args.seed
    if args.dataset is not None:
        config.data.active_datasets = [args.dataset]
    if args.gpu and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ── setup ──────────────────────────────────────────────────────
    config.setup()
    setup_logging(config)

    # Save the exact config used for reproducibility
    config.save(
        _resolve(config, "metrics_dir") / "config_used.yaml")

    # ── create LLM ─────────────────────────────────────────────────
    llm = LLMModule(config)

    logger.info(f"CrossGuard — mode={args.mode}")
    t0 = time.time()

    # ── dispatch ───────────────────────────────────────────────────
    mode = args.mode
    if mode == "eval":
        _eval_dataset(
            config, config.data.active_datasets[0],
            llm, train_lora=True)

    elif mode == "rq1":
        CrossDatasetExperiment(config, llm).run()

    elif mode == "rq2":
        CrossContractValueExperiment(config, llm).run()

    elif mode == "rq3":
        AblationExperiment(config, llm).run()

    elif mode == "rq4":
        ScalabilityExperiment(config, llm).run()

    elif mode == "rq5":
        ProtocolRoundsExperiment(config, llm).run()

    elif mode == "rq6":
        CaseStudyExperiment(config, llm).run()

    elif mode == "rq7":
        EfficiencyExperiment(config, llm).run()

    elif mode == "figures":
        generate_figures(config)

    elif mode == "tables":
        generate_latex_tables(config)

    elif mode == "all":
        CrossDatasetExperiment(config, llm).run()
        CrossContractValueExperiment(config, llm).run()
        AblationExperiment(config, llm).run()
        ScalabilityExperiment(config, llm).run()
        ProtocolRoundsExperiment(config, llm).run()
        CaseStudyExperiment(config, llm).run()
        EfficiencyExperiment(config, llm).run()
        generate_figures(config)
        generate_latex_tables(config)

    elapsed = time.time() - t0
    results_dir = config.paths.resolve("metrics_dir")
    logger.info(
        f"Done in {elapsed / 3600:.2f}h. "
        f"Results -> {results_dir}")
    print(
        f"\nDone in {elapsed / 3600:.2f}h. "
        f"Results -> {results_dir}")


if __name__ == "__main__":
    main()
