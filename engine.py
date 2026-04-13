"""
CrossGuard orchestration engine.

Components:
  CrossGuardMetrics    evaluation metrics including CC-Recall
  CrossGuardPipeline   three-phase DApp analysis pipeline
  AgentTrainer         LoRA fine-tuning with prompt-masked loss
  BaselineRunner       ablation baselines (single-agent, per-contract)
  ReportedBaselines    cited numbers from published papers
  CrossGuardEvaluator  end-to-end evaluation with case studies

Author : [Anonymous for double-blind review]
Target : ACM Transactions on Software Engineering and Methodology
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             roc_auc_score,
                             average_precision_score)

from .decomposer import DAppDecomposer
from .agents import (AgentPool, LLMModule, ContractAgent,
                     LocalFinding)
from .protocol import (CrossContractReasoningProtocol,
                       VulnerabilitySynthesiser,
                       CaseStudyReport)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════

class CrossGuardMetrics:
    """Accumulates per-DApp predictions and computes evaluation metrics.

    Metrics computed (SS4.4):
      Standard:   accuracy, precision, recall, F1, AUROC, AUPRC
      Cross:      cross_contract_recall (CC-Rec), cross_ratio (CC-Rat)
      Source:     llm_findings, pattern_findings
      Efficiency: avg_time_s, median_time_s, p95_time_s
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.preds: List[int] = []
        self.targets: List[int] = []
        self.scores: List[float] = []
        self.n_local: List[int] = []
        self.n_cross: List[int] = []
        self.n_contracts: List[int] = []
        self.n_interactions: List[int] = []
        self.times: List[float] = []
        # These lists MUST stay index-aligned with preds/targets.
        # Always append, even when the list is empty.
        self.all_findings: List[List[Dict]] = []
        self.all_vulns: List[List[Dict]] = []
        self.all_cross_gt: List[List[Dict]] = []

    def update(self, pred: int, target: int, score: float,
               n_loc: int, n_cr: int, n_con: int, n_int: int,
               elapsed: float,
               findings: Optional[List[Dict]] = None,
               vulns: Optional[List[Dict]] = None,
               cross_gt: Optional[List[Dict]] = None):
        """Record results for one DApp."""
        self.preds.append(pred)
        self.targets.append(target)
        self.scores.append(score)
        self.n_local.append(n_loc)
        self.n_cross.append(n_cr)
        self.n_contracts.append(n_con)
        self.n_interactions.append(n_int)
        self.times.append(elapsed)
        # Always append to maintain index alignment — an empty list
        # is semantically different from a missing entry.
        self.all_findings.append(findings if findings is not None
                                 else [])
        self.all_vulns.append(vulns if vulns is not None else [])
        self.all_cross_gt.append(cross_gt if cross_gt is not None
                                 else [])

    def compute(self) -> Dict[str, Any]:
        """Compute all metrics from accumulated predictions."""
        y = np.array(self.targets)
        yh = np.array(self.preds)
        ys = np.array(self.scores)

        m: Dict[str, Any] = {
            "n_dapps": len(y),
            "accuracy": float(accuracy_score(y, yh)),
            "precision": float(precision_score(
                y, yh, zero_division=0)),
            "recall": float(recall_score(
                y, yh, zero_division=0)),
            "f1_score": float(f1_score(
                y, yh, zero_division=0)),
        }

        # AUROC / AUPRC: require both classes present
        if len(np.unique(y)) >= 2:
            try:
                m["auroc"] = float(roc_auc_score(y, ys))
                m["auprc"] = float(
                    average_precision_score(y, ys))
            except ValueError:
                m["auroc"] = 0.0
                m["auprc"] = 0.0
        else:
            m["auroc"] = 0.0
            m["auprc"] = 0.0

        # ── cross-contract metrics ─────────────────────────────────
        m["total_local"] = sum(self.n_local)
        m["total_cross"] = sum(self.n_cross)
        total_findings = m["total_local"] + m["total_cross"]
        m["cross_ratio"] = (float(m["total_cross"])
                            / max(1, total_findings))

        m["avg_contracts"] = float(np.mean(self.n_contracts))
        m["avg_interactions"] = float(np.mean(self.n_interactions))

        # CC-Rec (SS4.4): among DApps with cross-contract indicators,
        # fraction where CrossGuard produced >= 1 cross-contract
        # finding AND correctly predicted vulnerable.
        cc_dapp_indices = [
            i for i in range(len(self.all_cross_gt))
            if self.all_cross_gt[i]]
        if cc_dapp_indices:
            detected = sum(
                1 for i in cc_dapp_indices
                if self.n_cross[i] > 0 and self.preds[i] == 1)
            m["cross_contract_recall"] = float(
                detected / len(cc_dapp_indices))
            m["n_cross_gt_dapps"] = len(cc_dapp_indices)
        else:
            m["cross_contract_recall"] = 0.0
            m["n_cross_gt_dapps"] = 0

        # ── source breakdown (Table IX) ────────────────────────────
        llm_count = sum(
            1 for fs in self.all_findings
            for f in fs if f.get("source") == "llm")
        pat_count = sum(
            1 for fs in self.all_findings
            for f in fs if f.get("source") == "pattern")
        local_count = sum(
            1 for fs in self.all_findings
            for f in fs if f.get("source") == "local")
        m["llm_findings"] = llm_count
        m["pattern_findings"] = pat_count
        m["local_findings"] = local_count

        # ── efficiency ─────────────────────────────────────────────
        m["avg_time_s"] = float(np.mean(self.times))
        m["median_time_s"] = float(np.median(self.times))
        if self.times:
            m["p95_time_s"] = float(np.percentile(self.times, 95))
        else:
            m["p95_time_s"] = 0.0

        return m

    # ── statistical tests (SS4.4) ──────────────────────────────────

    def bootstrap_ci(self, n_boot: int = 1000,
                     seed: int = 42,
                     ) -> Dict[str, Tuple[float, float]]:
        """Bootstrap 95% confidence intervals (SS4.4).

        Parameters read from config.reproducibility if available:
          num_bootstrap_samples (default 1000)
        """
        rng = np.random.RandomState(seed)
        y = np.array(self.targets)
        yh = np.array(self.preds)
        n = len(y)
        if n == 0:
            return {}

        results: Dict[str, List[float]] = defaultdict(list)
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            if len(np.unique(y[idx])) < 2:
                continue
            results["accuracy"].append(
                accuracy_score(y[idx], yh[idx]))
            results["precision"].append(
                precision_score(y[idx], yh[idx],
                                zero_division=0))
            results["recall"].append(
                recall_score(y[idx], yh[idx],
                             zero_division=0))
            results["f1_score"].append(
                f1_score(y[idx], yh[idx], zero_division=0))

        return {k: (float(np.percentile(v, 2.5)),
                     float(np.percentile(v, 97.5)))
                for k, v in results.items() if v}

    def permutation_test(self, other_preds: List[int],
                         n_perm: int = 10000,
                         seed: int = 42,
                         ) -> Dict[str, float]:
        """Two-sided permutation test p-values (SS4.4).

        Compares self.preds (CrossGuard) against *other_preds*
        (e.g., per-contract baseline) using the same targets.

        Paper: "two-sided p-values obtained via permutation testing
        (10,000 permutations)"
        """
        rng = np.random.RandomState(seed)
        y = np.array(self.targets)
        yh_a = np.array(self.preds)
        yh_b = np.array(other_preds)
        n = len(y)
        if n == 0:
            return {}

        def _f1(yh):
            return f1_score(y, yh, zero_division=0)

        def _prec(yh):
            return precision_score(y, yh, zero_division=0)

        def _rec(yh):
            return recall_score(y, yh, zero_division=0)

        def _acc(yh):
            return accuracy_score(y, yh)

        metrics = {
            "accuracy": _acc,
            "precision": _prec,
            "recall": _rec,
            "f1_score": _f1,
        }

        p_values: Dict[str, float] = {}
        for metric_name, metric_fn in metrics.items():
            observed = abs(metric_fn(yh_a) - metric_fn(yh_b))
            count = 0
            pooled = np.stack([yh_a, yh_b], axis=0)
            for _ in range(n_perm):
                # For each sample, randomly swap A and B predictions
                swap = rng.randint(0, 2, size=n)
                perm_a = np.where(swap == 0, pooled[0], pooled[1])
                perm_b = np.where(swap == 0, pooled[1], pooled[0])
                perm_diff = abs(metric_fn(perm_a) - metric_fn(perm_b))
                if perm_diff >= observed:
                    count += 1
            p_values[metric_name] = (count + 1) / (n_perm + 1)

        return p_values


# ═══════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════

class CrossGuardPipeline:
    """Three-phase DApp vulnerability analysis pipeline.

    Phase 1: DApp decomposition + per-contract local analysis
    Phase 2: Cross-contract reasoning along interaction graph
    Phase 3: Vulnerability synthesis and ranking
    """

    def __init__(self, config, llm: Optional[LLMModule] = None):
        self.config = config
        self.decomposer = DAppDecomposer(config)
        self.synthesiser = VulnerabilitySynthesiser(config, llm)
        self.llm = llm

    def analyse_dapp(self, dapp_files: Dict[str, str],
                     dapp_id: str = "",
                     generate_graph_fig: bool = False,
                     ) -> Dict[str, Any]:
        """Run the full three-phase pipeline on one DApp.

        Returns a dict with: prediction, score, report, timing,
        graph stats, convergence data.
        """
        t0 = time.time()

        # ── Phase 1: decomposition + local analysis ────────────────
        graph = self.decomposer.decompose(dapp_files, dapp_id)

        if (generate_graph_fig
                and self.config.decomposer.generate_graph_figures):
            fig_dir = self.config.paths.resolve("figures_dir")
            fig_path = fig_dir / f"graph_{dapp_id}.pdf"
            graph.visualise(
                str(fig_path),
                self.config.decomposer.graph_layout)

        pool = AgentPool(graph, self.config, self.llm)
        summaries = pool.run_local_analysis()

        # ── Phase 2: cross-contract reasoning ──────────────────────
        protocol = CrossContractReasoningProtocol(
            self.config, graph, pool)
        cross_vulns = protocol.run(summaries)

        # ── Phase 3: synthesis ─────────────────────────────────────
        report = self.synthesiser.synthesise(summaries, cross_vulns)
        elapsed = time.time() - t0

        # ── DApp-level prediction ──────────────────────────────────
        max_score = max(
            (f["score"] for f in report), default=0.0)
        pred = (1 if max_score
                >= self.config.protocol.vulnerability_score_threshold
                else 0)

        return {
            "dapp_id": dapp_id,
            "prediction": pred,
            "score": max_score,
            "report": report,
            "n_local": sum(len(s.local_findings)
                           for s in summaries.values()),
            "n_cross": len(cross_vulns),
            "n_contracts": graph.num_contracts,
            "n_interactions": graph.num_interactions,
            "time_s": elapsed,
            "graph_summary": graph.summary(),
            "convergence": protocol.get_convergence_data(),
            "n_messages": len(protocol.message_history),
        }


# ═══════════════════════════════════════════════════════════════════════
# LoRA fine-tuning
# ═══════════════════════════════════════════════════════════════════════

class AgentTrainer:
    """Fine-tune the shared LLM backbone using LoRA (SS3.5).

    Training setup from the paper:
      - AdamW, lr=2e-5, weight_decay=0.01, cosine warmup schedule
      - Gradient accumulation: 8 steps (effective batch = 32)
      - Gradient clip: max norm 1.0
      - Early stopping: patience 10, monitors validation loss
      - Up to 10 epochs; best checkpoint restored

    Supervision: for each contract with known vulnerabilities, a
    (prompt, target) pair is created.  Loss is computed ONLY over the
    target tokens; prompt tokens are masked with label=-100 so the
    model learns to generate correct vulnerability JSON, not to
    predict the prompt itself.
    """

    def __init__(self, config, llm: LLMModule):
        self.config = config
        self.llm = llm

    def train(self, train_loader,
              val_loader=None) -> Dict[str, Any]:
        """Fine-tune LoRA parameters with early stopping.

        Returns training history dict with per-epoch losses.
        """
        self.llm._load_model()
        if self.llm._model is None:
            logger.warning(
                "LLM not available — skipping fine-tuning")
            return {}

        params = self.llm.get_trainable_parameters()
        if not params:
            logger.warning(
                "No trainable parameters — skipping fine-tuning")
            return {}

        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.training.lora_finetune_lr,
            weight_decay=self.config.training.weight_decay)

        # Cosine learning rate schedule with warmup
        n_epochs = self.config.training.lora_finetune_epochs
        warmup_ratio = self.config.training.lr_warmup_ratio
        # Estimate total steps for scheduler
        n_batches_per_epoch = max(1, len(train_loader))
        total_steps = (n_epochs * n_batches_per_epoch
                       // self.config.training
                       .gradient_accumulation_steps)
        warmup_steps = max(1, int(total_steps * warmup_ratio))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=_cosine_warmup_lambda(
                warmup_steps, total_steps))

        patience = self.config.training.early_stopping_patience

        logger.info(
            f"LoRA fine-tuning: "
            f"{sum(p.numel() for p in params):,} parameters, "
            f"{n_epochs} epochs, patience={patience}, "
            f"warmup={warmup_steps}/{total_steps} steps")

        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0

        for epoch in range(n_epochs):
            # ── training phase ─────────────────────────────────────
            train_loss, n_train = self._run_epoch(
                train_loader, optimizer, params, scheduler)
            history["train_loss"].append(train_loss)

            # ── validation phase ───────────────────────────────────
            val_loss = float("inf")
            if val_loader is not None:
                val_loss, _ = self._run_epoch(
                    val_loader, optimizer=None,
                    params=None, scheduler=None)
                history["val_loss"].append(val_loss)

            logger.info(
                f"  LoRA epoch {epoch + 1}/{n_epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f} "
                f"({n_train} samples)")

            # ── early stopping ─────────────────────────────────────
            metric = (val_loss if val_loader is not None
                      else train_loss)
            if metric < best_val_loss - 1e-4:
                best_val_loss = metric
                patience_counter = 0
                if self.llm._model is not None:
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in
                        self.llm._model.state_dict().items()
                        if "lora" in k.lower()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        f"  Early stopping at epoch {epoch + 1} "
                        f"(best_loss={best_val_loss:.4f})")
                    break

        # ── restore best checkpoint ────────────────────────────────
        if best_state and self.llm._model is not None:
            current = self.llm._model.state_dict()
            current.update(best_state)
            self.llm._model.load_state_dict(
                current, strict=False)
            logger.info(
                f"  Restored best checkpoint "
                f"(loss={best_val_loss:.4f})")

        return history

    def _run_epoch(self, loader, optimizer=None, params=None,
                   scheduler=None) -> Tuple[float, int]:
        """Run one training or validation epoch.

        Key fix: the loss is computed ONLY over target tokens.
        Prompt tokens are masked with label=-100 so the model
        does not learn to predict the prompt (which would
        produce a model that memorises prompt structure rather
        than learning vulnerability analysis).
        """
        is_train = optimizer is not None
        model = self.llm._model
        tokenizer = self.llm._tokenizer

        if model is None or tokenizer is None:
            return 0.0, 0

        if is_train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        n_samples = 0
        accum_steps = self.config.training.gradient_accumulation_steps

        if is_train and optimizer is not None:
            optimizer.zero_grad()

        desc = "LoRA-train" if is_train else "LoRA-val"
        for batch in tqdm(loader, desc=desc, leave=False):
            for sample in batch:
                for fname, code in sample["files"].items():
                    if not code or len(code) < 50:
                        continue

                    # ── construct prompt + target ──────────────────
                    prompt, target = self._build_training_pair(
                        code, sample)
                    if not target:
                        continue

                    try:
                        loss = self._compute_masked_loss(
                            model, tokenizer, prompt, target,
                            is_train)
                    except Exception as exc:
                        logger.debug(
                            f"{'Train' if is_train else 'Val'} "
                            f"step failed: {exc}")
                        continue

                    if loss is None:
                        continue

                    total_loss += loss.item()
                    n_samples += 1

                    if is_train:
                        scaled = loss / accum_steps
                        scaled.backward()
                        if n_samples % accum_steps == 0:
                            torch.nn.utils.clip_grad_norm_(
                                params,
                                self.config.training
                                .gradient_clip_value)
                            optimizer.step()
                            if scheduler is not None:
                                scheduler.step()
                            optimizer.zero_grad()

        # Flush remaining gradients
        if is_train and n_samples % accum_steps != 0:
            if params is not None:
                torch.nn.utils.clip_grad_norm_(
                    params,
                    self.config.training.gradient_clip_value)
            if optimizer is not None:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

        return total_loss / max(1, n_samples), n_samples

    def _build_training_pair(self, code: str,
                             sample: Dict[str, Any],
                             ) -> Tuple[str, str]:
        """Construct (prompt, target_json) for one training contract.

        The prompt follows the same template used at inference time
        (SS3.2).  The target is the ground-truth vulnerability JSON.
        """
        has_vuln = sample["label"] == 1
        vuln_types = [v.get("type", "unknown")
                      for v in sample["vulnerabilities"]]

        prompt = (
            "You are a smart contract security auditor. "
            "Analyse the following Solidity contract for "
            "vulnerabilities.\n\n"
            f"```solidity\n{code[:2000]}\n```\n\n"
            "Respond ONLY with a JSON object:\n")

        if has_vuln:
            findings = [
                {"function": "", "type": vt, "confidence": 0.9}
                for vt in vuln_types[:3]]
            target = json.dumps({
                "findings": findings,
                "risk_score": 0.8})
        else:
            target = json.dumps({
                "findings": [],
                "risk_score": 0.1})

        return prompt, target

    def _compute_masked_loss(self, model, tokenizer,
                             prompt: str, target: str,
                             is_train: bool,
                             ) -> Optional[torch.Tensor]:
        """Compute causal LM loss with prompt tokens masked.

        The key difference from the original code: we tokenise
        prompt and target separately, then set labels=-100 for
        all prompt token positions.  This ensures the model is
        trained ONLY on generating the target JSON, not on
        predicting the prompt text.
        """
        max_len = self.config.agent.llm_max_length

        prompt_ids = tokenizer(
            prompt, add_special_tokens=True,
            truncation=True, max_length=max_len // 2,
            return_tensors="pt")["input_ids"]

        target_ids = tokenizer(
            target, add_special_tokens=False,
            truncation=True,
            max_length=max_len - prompt_ids.shape[1],
            return_tensors="pt")["input_ids"]

        input_ids = torch.cat(
            [prompt_ids, target_ids], dim=1)

        # Labels: -100 for prompt tokens (ignored by cross-entropy),
        # actual token ids for target tokens
        labels = input_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100

        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)

        with torch.set_grad_enabled(is_train):
            outputs = model(
                input_ids=input_ids,
                labels=labels)
            return outputs.loss


# ═══════════════════════════════════════════════════════════════════════
# Ablation baselines
# ═══════════════════════════════════════════════════════════════════════

class BaselineRunner:
    """Run ablation baselines for fair comparison (SS4.2).

    IMPORTANT METHODOLOGICAL NOTE:
    These are NOT reimplementations of SmartAuditFlow or MANDO-LLM.
    They are ablation variants that test CrossGuard's architecture
    under conditions *analogous* to prior paradigms:

      single_agent_llm      one LLM analyses the whole DApp at once
      per_contract_no_cross  per-contract agents, no Phase 2

    The paper presents these as ablation baselines (SS4.2) and
    separately cites reported numbers from original publications.
    """

    def __init__(self, config, llm: Optional[LLMModule] = None):
        self.config = config
        self.llm = llm

    def run_single_agent_llm(self, dapp_files: Dict[str, str],
                             dapp_id: str = "",
                             ) -> Dict[str, Any]:
        """Ablation: single LLM call on concatenated DApp code.

        Concatenates up to 5 files (1 500 chars each) into one prompt.
        Analogous to the single-agent paradigm described in SS4.2.
        """
        all_code = "\n\n".join(
            f"// File: {fn}\n{code[:1500]}"
            for fn, code in list(dapp_files.items())[:5])

        if (self.llm and self.llm.is_available
                and not self.config.agent.use_fallback_mlp):
            prompt = (
                "Analyse this smart contract code for "
                "vulnerabilities:\n"
                f"```solidity\n{all_code[:4000]}\n```\n"
                "Respond with JSON: "
                '{"findings": [...], "risk_score": 0-1}')
            try:
                raw = self.llm.generate(prompt, 512)
                parsed = LLMModule.parse_json_response(raw)
                findings = parsed.get("findings", [])
                risk = float(parsed.get("risk_score", 0))
                threshold = (self.config.protocol
                             .vulnerability_score_threshold)
                return {
                    "pred": 1 if risk > threshold else 0,
                    "score": risk,
                    "n_findings": len(findings),
                    "method": "single_agent_llm"}
            except Exception as exc:
                logger.debug(
                    f"Single-agent baseline failed: {exc}")

        return {"pred": 0, "score": 0.0,
                "n_findings": 0,
                "method": "single_agent_llm"}

    def run_per_contract_no_cross(
            self, dapp_files: Dict[str, str],
            dapp_id: str = "") -> Dict[str, Any]:
        """Ablation: per-contract agents, no cross-contract reasoning.

        Each contract gets its own agent for local analysis, but
        agents do NOT communicate.  Analogous to the isolated-contract
        paradigm described in SS4.2.
        """
        all_findings: List[LocalFinding] = []
        max_risk = 0.0
        decomposer = DAppDecomposer(self.config)
        graph = decomposer.decompose(dapp_files, dapp_id)

        for cname, ci in graph.contracts.items():
            agent = ContractAgent(
                f"bl_{cname}", ci, self.llm, self.config)
            summary = agent.run_local_analysis()
            all_findings.extend(summary.local_findings)
            max_risk = max(max_risk, summary.risk_score)

        threshold = self.config.protocol.vulnerability_score_threshold
        return {
            "pred": 1 if max_risk > threshold else 0,
            "score": max_risk,
            "n_findings": len(all_findings),
            "method": "per_contract_no_cross"}

    def run_slither(self, source_code: str,
                    ) -> Dict[str, Any]:
        """Run Slither on a single contract source file.

        Used for per-file baseline comparison on DAppSCAN.
        Slither is invoked as an external process.
        """
        with tempfile.NamedTemporaryFile(
                suffix=".sol", mode="w", delete=False) as f:
            f.write(source_code)
            tmp = f.name
        try:
            result = subprocess.run(
                [self.config.baselines.slither_path,
                 tmp, "--json", "-"],
                capture_output=True, text=True,
                timeout=self.config.baselines.tool_timeout)
            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    detectors = (data.get("results", {})
                                 .get("detectors", []))
                    return {
                        "pred": 1 if detectors else 0,
                        "n_findings": len(detectors),
                        "method": "slither"}
                except json.JSONDecodeError:
                    pass
            return {"pred": 0, "n_findings": 0,
                    "method": "slither"}
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return {"pred": 0, "n_findings": 0,
                    "method": "slither",
                    "error": str(exc)}
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def run_mythril(self, source_code: str,
                    ) -> Dict[str, Any]:
        """Run Mythril on a single contract source file."""
        with tempfile.NamedTemporaryFile(
                suffix=".sol", mode="w", delete=False) as f:
            f.write(source_code)
            tmp = f.name
        try:
            result = subprocess.run(
                [self.config.baselines.mythril_path,
                 "analyze", tmp, "-o", "json"],
                capture_output=True, text=True,
                timeout=self.config.baselines.tool_timeout)
            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    issues = data.get("issues", [])
                    return {
                        "pred": 1 if issues else 0,
                        "n_findings": len(issues),
                        "method": "mythril"}
                except json.JSONDecodeError:
                    pass
            return {"pred": 0, "n_findings": 0,
                    "method": "mythril"}
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return {"pred": 0, "n_findings": 0,
                    "method": "mythril",
                    "error": str(exc)}
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════
# Reported baseline numbers
# ═══════════════════════════════════════════════════════════════════════

class ReportedBaselines:
    """Cited numbers from published baseline papers.

    These are NOT our own measurements.  They are copied directly
    from the original publications for comparison tables.

    In the paper, reported numbers are marked with dagger (SS5.1):
    "cited numbers are marked with a dagger in all tables and are
    never presented as our own measurements."
    """

    # ESC: reported by original papers
    ESC = {
        "Slither": {
            "accuracy": 77.12, "precision": 71.80,
            "recall": 73.45, "f1": 71.23,
            "source": "Feist et al., WETSEB 2019",
            "cite_key": "feist2019slither"},
        "Mythril": {
            "accuracy": 68.34, "precision": 63.17,
            "recall": 69.21, "f1": 65.89,
            "source": "Mueller, 2017",
            "cite_key": "mueller2017mythril"},
        "DR-GCN": {
            "accuracy": 81.47, "precision": 72.36,
            "recall": 80.89, "f1": 76.39,
            "source": "Zhuang et al., IJCAI 2020",
            "cite_key": "zhuang2020gnn",
            "task": "reentrancy"},
        "TMP": {
            "accuracy": 84.48, "precision": 74.06,
            "recall": 82.63, "f1": 78.11,
            "source": "Zhuang et al., IJCAI 2020",
            "cite_key": "zhuang2020gnn",
            "task": "reentrancy"},
        "AME": {
            "accuracy": 90.19, "precision": 86.25,
            "recall": 89.69, "f1": 87.94,
            "source": "Liu et al., IJCAI 2021",
            "cite_key": "liu2021ame",
            "task": "reentrancy"},
        "CGE": {
            "accuracy": 89.15, "precision": 85.24,
            "recall": 87.62, "f1": 86.41,
            "source": "Liu et al., TKDE 2023",
            "cite_key": "liu2023cge",
            "task": "reentrancy"},
        "EFEVD": {
            "accuracy": 89.53, "precision": 87.72,
            "recall": 92.82, "f1": 91.18,
            "source": "Jiang et al., IJCAI 2024",
            "cite_key": "jiang2024efevd",
            "task": "contract-level"},
    }

    # SMS: reported by original papers
    SMS = {
        "SMS_reentrancy": {
            "accuracy": 83.85, "precision": 79.46,
            "recall": 77.48, "f1": 78.46,
            "source": "Qian et al., WWW 2023",
            "cite_key": "qian2023sms"},
        "SMS_timestamp": {
            "accuracy": 89.77, "precision": 89.15,
            "recall": 91.09, "f1": 90.11,
            "source": "Qian et al., WWW 2023",
            "cite_key": "qian2023sms"},
        "SMS_overflow": {
            "accuracy": 79.36, "precision": 78.14,
            "recall": 72.98, "f1": 75.47,
            "source": "Qian et al., WWW 2023",
            "cite_key": "qian2023sms"},
        "SMS_delegatecall": {
            "accuracy": 78.82, "precision": 76.97,
            "recall": 73.69, "f1": 75.29,
            "source": "Qian et al., WWW 2023",
            "cite_key": "qian2023sms"},
        "Agent4Vul_reentrancy": {
            "accuracy": 92.73, "precision": 92.68,
            "recall": 92.73, "f1": 92.53,
            "source": "Jie et al., SCIS 2025",
            "cite_key": "jie2025agent4vul"},
        "Agent4Vul_timestamp": {
            "accuracy": 98.57, "precision": 98.62,
            "recall": 98.57, "f1": 98.58,
            "source": "Jie et al., SCIS 2025",
            "cite_key": "jie2025agent4vul"},
        "Agent4Vul_overflow": {
            "accuracy": 98.18, "precision": 98.29,
            "recall": 98.18, "f1": 98.20,
            "source": "Jie et al., SCIS 2025",
            "cite_key": "jie2025agent4vul"},
        "Agent4Vul_delegatecall": {
            "accuracy": 97.50, "precision": 97.58,
            "recall": 97.50, "f1": 97.45,
            "source": "Jie et al., SCIS 2025",
            "cite_key": "jie2025agent4vul"},
    }

    # DAppSCAN: MANDO-LLM reports on mixed Dataset B, not pure DAppSCAN
    DAPPSCAN_MIXED = {
        "MANDO-LLM_reentrancy": {
            "buggy_f1": 89.55, "macro_f1": 89.22,
            "source": "Nguyen et al., TOSEM 2025",
            "cite_key": "nguyen2025mandollm",
            "note": "Dataset B = DAppSCAN + SolidiFI + "
                    "SmartBugs Wild (mixed, not pure DAppSCAN)"},
        "MANDO-LLM_timestamp": {
            "buggy_f1": 94.12, "macro_f1": 94.28,
            "source": "Nguyen et al., TOSEM 2025",
            "cite_key": "nguyen2025mandollm",
            "note": "Dataset B (mixed)"},
    }

    # Non-dataset-specific
    LLM_SMARTAUDIT = {
        "common_vuln_set": {
            "accuracy": 98.0, "precision": 99.0,
            "recall": 98.0, "f1": 98.5,
            "source": "Wei et al., TSE 2025",
            "cite_key": "wei2025multiagent",
            "note": "GPT-4o TA mode on 110-contract "
                    "Common-Vulnerability Set"},
        "real_world_set": {
            "accuracy": 47.6,
            "source": "Wei et al., TSE 2025",
            "cite_key": "wei2025multiagent",
            "note": "TA mode on 6,454 real-world contracts"},
    }

    @classmethod
    def get_for_dataset(cls, dataset: str) -> Dict[str, Dict]:
        """Get reported baseline numbers for a specific dataset."""
        mapping = {
            "esc": cls.ESC,
            "sms": cls.SMS,
            "dappscan": cls.DAPPSCAN_MIXED,
        }
        return mapping.get(dataset, {})

    @classmethod
    def generate_comparison_table(
            cls, crossguard_results: Dict[str, Any],
            dataset: str) -> List[Dict[str, Any]]:
        """Combine CrossGuard results with reported baselines.

        Returns a list of row dicts for Table I.  Each row has
        'is_ours' to distinguish measured from cited numbers.
        """
        reported = cls.get_for_dataset(dataset)
        rows: List[Dict[str, Any]] = []

        for name, nums in reported.items():
            rows.append({
                "method": name,
                "accuracy": nums.get("accuracy", "—"),
                "precision": nums.get("precision", "—"),
                "recall": nums.get("recall", "—"),
                "f1": nums.get("f1",
                               nums.get("buggy_f1", "—")),
                "source": nums.get("source", ""),
                "cite_key": nums.get("cite_key", ""),
                "is_ours": False})

        m = crossguard_results.get("metrics", {})
        rows.append({
            "method": "CrossGuard",
            "accuracy": f"{m.get('accuracy', 0) * 100:.2f}",
            "precision": f"{m.get('precision', 0) * 100:.2f}",
            "recall": f"{m.get('recall', 0) * 100:.2f}",
            "f1": f"{m.get('f1_score', 0) * 100:.2f}",
            "source": "Ours",
            "cite_key": "",
            "is_ours": True})

        return rows


# ═══════════════════════════════════════════════════════════════════════
# Evaluator
# ═══════════════════════════════════════════════════════════════════════

class CrossGuardEvaluator:
    """End-to-end evaluation with metrics, baselines, and case studies.

    Orchestrates the full evaluation workflow:
      1. Run CrossGuard pipeline on test set
      2. Compute classification and cross-contract metrics
      3. Run ablation baselines for comparison
      4. Generate case studies for qualitative analysis (RQ6)
    """

    def __init__(self, config, pipeline: CrossGuardPipeline,
                 llm: Optional[LLMModule] = None):
        self.config = config
        self.pipeline = pipeline
        self.metrics = CrossGuardMetrics()
        self.baseline_runner = BaselineRunner(config, llm)

    def evaluate(self, test_loader,
                 ) -> Dict[str, Any]:
        """Evaluate CrossGuard on a test set.

        Returns dict with metrics and confidence intervals.
        """
        self.metrics.reset()
        for batch in tqdm(test_loader, desc="Evaluating",
                          leave=False):
            for s in batch:
                r = self.pipeline.analyse_dapp(
                    s["files"], s["dapp_id"])
                self.metrics.update(
                    pred=r["prediction"],
                    target=s["label"],
                    score=r["score"],
                    n_loc=r["n_local"],
                    n_cr=r["n_cross"],
                    n_con=r["n_contracts"],
                    n_int=r["n_interactions"],
                    elapsed=r["time_s"],
                    findings=r["report"],
                    vulns=s["vulnerabilities"],
                    cross_gt=s.get("cross_contract_gt", []))

        m = self.metrics.compute()
        n_boot = self.config.reproducibility.num_bootstrap_samples
        ci = self.metrics.bootstrap_ci(
            n_boot=n_boot,
            seed=self.config.reproducibility.seed)
        self._print_results(m, ci)
        return {"metrics": m, "confidence_intervals": ci}

    def evaluate_baselines(self, test_loader,
                           ) -> Dict[str, Dict[str, float]]:
        """Run ablation baselines and collect their predictions.

        Returns per-baseline metrics dict.  Also stores baseline
        predictions for permutation testing against CrossGuard.
        """
        logger.info("Running ablation baselines ...")
        bl_data: Dict[str, Dict[str, List]] = defaultdict(
            lambda: {"preds": [], "targets": []})

        for batch in tqdm(test_loader, desc="Baselines",
                          leave=False):
            for s in batch:
                t = s["label"]
                if self.config.baselines.run_smartauditflow:
                    r = self.baseline_runner.run_single_agent_llm(
                        s["files"], s["dapp_id"])
                    bl_data["single_agent_llm"]["preds"].append(
                        r["pred"])
                    bl_data["single_agent_llm"]["targets"].append(t)

                if self.config.baselines.run_mando_llm:
                    r = (self.baseline_runner
                         .run_per_contract_no_cross(
                             s["files"], s["dapp_id"]))
                    bl_data["per_contract_no_cross"]["preds"]\
                        .append(r["pred"])
                    bl_data["per_contract_no_cross"]["targets"]\
                        .append(t)

        metrics: Dict[str, Dict[str, float]] = {}
        for bl_name, d in bl_data.items():
            y = np.array(d["targets"])
            yh = np.array(d["preds"])
            metrics[bl_name] = {
                "accuracy": float(accuracy_score(y, yh)),
                "precision": float(precision_score(
                    y, yh, zero_division=0)),
                "recall": float(recall_score(
                    y, yh, zero_division=0)),
                "f1_score": float(f1_score(
                    y, yh, zero_division=0))}
            logger.info(
                f"  {bl_name}: "
                f"F1={metrics[bl_name]['f1_score']:.4f}")

            # Permutation test: CrossGuard vs this baseline
            if len(self.metrics.preds) == len(d["preds"]):
                n_perm = (self.config.reproducibility
                          .num_permutation_tests)
                p_vals = self.metrics.permutation_test(
                    d["preds"],
                    n_perm=n_perm,
                    seed=self.config.reproducibility.seed)
                metrics[bl_name]["p_values"] = p_vals
                logger.info(
                    f"    p-values: {p_vals}")

        return metrics

    def generate_case_studies(self, test_loader,
                              n: int = 10,
                              ) -> List[CaseStudyReport]:
        """Generate structured case studies for RQ6 (SS5.7).

        Selects multi-contract DApps with known vulnerabilities
        where CrossGuard finds cross-contract findings.
        """
        cases: List[CaseStudyReport] = []

        for batch in test_loader:
            for s in batch:
                if (not s["vulnerabilities"]
                        or s["num_contracts"] < 2):
                    continue

                r = self.pipeline.analyse_dapp(
                    s["files"], s["dapp_id"],
                    generate_graph_fig=True)

                if r["n_cross"] <= 0:
                    continue

                cross_findings = [
                    f for f in r["report"]
                    if f.get("is_cross_contract")]
                local_findings = [
                    f for f in r["report"]
                    if not f.get("is_cross_contract")]

                # Count local findings per contract
                local_per_contract: Dict[str, int] = defaultdict(int)
                for f in local_findings:
                    for c in f.get("contracts", []):
                        local_per_contract[c] += 1

                cs = CaseStudyReport(
                    dapp_id=s["dapp_id"],
                    num_contracts=s["num_contracts"],
                    graph_summary=r["graph_summary"],
                    cross_contract_findings=cross_findings[:5],
                    local_findings_per_contract=dict(
                        local_per_contract),
                    convergence_rounds=r["convergence"].get(
                        "n_rounds_used", 0),
                    what_single_contract_would_miss=[
                        f"{f['type']}: {f['description']}"
                        for f in cross_findings[:3]],
                    ground_truth=s["vulnerabilities"][:5])
                cases.append(cs)

                if len(cases) >= n:
                    break
            if len(cases) >= n:
                break

        logger.info(f"Generated {len(cases)} case studies")
        return cases

    def _print_results(self, m: Dict[str, Any],
                       ci: Dict[str, Tuple[float, float]]):
        """Print evaluation results to stdout."""
        lines = [
            "",
            "=" * 65,
            "  CrossGuard EVALUATION",
            "=" * 65,
        ]
        for k in ("accuracy", "precision", "recall",
                   "f1_score", "auroc"):
            lo, hi = ci.get(k, (0, 0))
            lines.append(
                f"  {k:28s}: {m.get(k, 0):.4f}  "
                f"[{lo:.4f}, {hi:.4f}]")
        lines.append("")
        lines.append("  Cross-Contract Metrics:")
        for k in ("total_local", "total_cross", "cross_ratio",
                   "cross_contract_recall", "n_cross_gt_dapps",
                   "llm_findings", "pattern_findings",
                   "avg_contracts", "avg_time_s",
                   "median_time_s", "p95_time_s"):
            v = m.get(k, 0)
            if isinstance(v, float):
                lines.append(f"    {k:28s}: {v:.4f}")
            else:
                lines.append(f"    {k:28s}: {v}")
        lines.append("=" * 65)
        print("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════
# Module-level helpers
# ═══════════════════════════════════════════════════════════════════════

def _cosine_warmup_lambda(warmup_steps: int,
                          total_steps: int):
    """Return a lr_lambda for cosine schedule with linear warmup.

    Paper (SS3.5): "cosine schedule with a warmup ratio of 0.1"
    """
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(
            1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(
            math.pi * progress)))

    return lr_lambda
