"""
Configuration management for CrossGuard.

Three-phase multi-agent cross-contract vulnerability detection::

    Phase 1 (Local):   Agent_i analyses Contract_i -> findings F_i
    Phase 2 (Cross):   Agents reason along interaction graph -> paths P
    Phase 3 (Synth):   Aggregate F + P -> ranked vulnerability report

All default values match those reported in the paper (SS4.3).

Author : [Anonymous for double-blind review]
Target : ACM Transactions on Software Engineering and Methodology
"""

import json
import logging
import math
import os
import random
import warnings
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when a configuration value is invalid."""
    pass


# ═══════════════════════════════════════════════════════════════════════
# Dataclass sub-configs
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AgentConfig:
    """Per-contract LLM agent configuration (SS3.2).

    Each ContractAgent wraps a shared LLM backbone (deepseek-coder)
    fine-tuned with QLoRA.  One agent is assigned per contract.
    """
    # ── LLM backbone ───────────────────────────────────────────────
    llm_model: str = "deepseek-ai/deepseek-coder-6.7b-instruct"
    llm_max_length: int = 4096

    # ── QLoRA (SS3.5) ──────────────────────────────────────────────
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    quantisation_bits: int = 4       # 4-bit NormalFloat

    # ── Agent behaviour ────────────────────────────────────────────
    max_agents_per_dapp: int = 20    # agent budget cap
    agent_temperature: float = 0.1   # sampling temperature
    local_analysis_max_tokens: int = 1024
    cross_contract_max_tokens: int = 768
    cross_reasoning_retries: int = 2 # retries on JSON parse failure

    # ── LLM vs fallback ───────────────────────────────────────────
    use_llm_cross_reasoning: bool = True   # LLM for Phase 2
    use_fallback_mlp: bool = False         # MLP ablation (RQ3)
    fallback_hidden_dim: int = 256

    def validate(self):
        if self.lora_rank <= 0:
            raise ConfigError("lora_rank must be positive")
        if self.lora_alpha <= 0:
            raise ConfigError("lora_alpha must be positive")
        if self.max_agents_per_dapp < 1:
            raise ConfigError("max_agents_per_dapp must be >= 1")
        if not (0.0 <= self.agent_temperature <= 2.0):
            raise ConfigError(
                f"agent_temperature={self.agent_temperature} "
                f"outside [0, 2]")
        if self.local_analysis_max_tokens < 0:
            raise ConfigError("local_analysis_max_tokens must be >= 0")
        if self.cross_reasoning_retries < 1:
            raise ConfigError("cross_reasoning_retries must be >= 1")


@dataclass
class ProtocolConfig:
    """Cross-contract reasoning protocol (SS3.3, Algorithm 1).

    Controls the iterative message-passing rounds, convergence
    tracking, vulnerability scoring, and multi-hop path detection.
    """
    num_reasoning_rounds: int = 3       # T in Algorithm 1
    message_max_tokens: int = 512
    vulnerability_score_threshold: float = 0.5  # Equation (1)
    max_path_length: int = 5            # multi-hop DFS limit
    convergence_threshold: float = 0.05 # theta in Algorithm 1

    # ── iterative refinement ───────────────────────────────────────
    use_iterative_refinement: bool = True
    track_message_similarity: bool = True
    similarity_method: str = "cosine"   # "cosine" or "jaccard"

    # ── cross-contract vulnerability families (SS3.3) ──────────────
    cross_contract_patterns: List[str] = field(default_factory=lambda: [
        "cross_reentrancy",
        "approval_abuse",
        "oracle_manipulation",
        "proxy_storage_collision",     # aligned with CrossContractPatterns
        "flash_loan_attack",
        "shared_state_corruption",
        "callback_chain",
        "privilege_escalation",
    ])

    def validate(self):
        if self.num_reasoning_rounds < 0:
            raise ConfigError("num_reasoning_rounds must be >= 0")
        if not (0.0 <= self.vulnerability_score_threshold <= 1.0):
            raise ConfigError(
                f"vulnerability_score_threshold="
                f"{self.vulnerability_score_threshold} outside [0, 1]")
        if not (0.0 < self.convergence_threshold < 1.0):
            raise ConfigError(
                f"convergence_threshold={self.convergence_threshold} "
                f"outside (0, 1)")
        if self.max_path_length < 2:
            raise ConfigError("max_path_length must be >= 2")
        if self.similarity_method not in ("cosine", "jaccard"):
            raise ConfigError(
                f"similarity_method must be 'cosine' or 'jaccard', "
                f"got '{self.similarity_method}'")


@dataclass
class DecomposerConfig:
    """DApp decomposition and interaction graph (SS3.2).

    Controls which of the six edge types are detected and the
    visualisation layout for paper figures.
    """
    max_contracts_per_dapp: int = 50

    # ── six interaction types ──────────────────────────────────────
    detect_inheritance: bool = True
    detect_external_calls: bool = True
    detect_interface_deps: bool = True
    detect_state_deps: bool = True
    detect_event_deps: bool = True
    detect_proxy_patterns: bool = True

    # ── graph visualisation ────────────────────────────────────────
    generate_graph_figures: bool = True
    graph_layout: str = "spring"   # "spring", "circular"

    def validate(self):
        if self.max_contracts_per_dapp < 1:
            raise ConfigError("max_contracts_per_dapp must be >= 1")


@dataclass
class TrainingConfig:
    """LoRA fine-tuning for the shared agent LLM backbone (SS3.5).

    The paper's stated training setup:
      - AdamW, lr=2e-5, weight_decay=0.01
      - Cosine schedule with warmup_ratio=0.1
      - Gradient accumulation: 8 steps (effective batch=32)
      - Gradient clip: 1.0
      - Mixed precision: float16
      - Early stopping: patience 10, monitors validation loss
      - Up to 10 LoRA fine-tuning epochs

    Fields marked [active] are consumed by AgentTrainer in engine.py.
    Fields marked [reserved] are declared for completeness but are not
    yet wired into the training loop; they will produce a warning if
    set to non-default values.
    """
    # ── active: consumed by AgentTrainer ───────────────────────────
    batch_size: int = 4                       # [active] DataLoader
    lora_finetune_epochs: int = 10            # [active] max epochs
    lora_finetune_lr: float = 2e-5            # [active] AdamW lr
    weight_decay: float = 0.01                # [active] AdamW
    gradient_accumulation_steps: int = 8      # [active]
    gradient_clip_value: float = 1.0          # [active]
    early_stopping_patience: int = 10         # [active] val-loss

    # ── reserved: not yet wired ────────────────────────────────────
    lr_scheduler: str = "cosine_warmup"       # [reserved]
    lr_warmup_ratio: float = 0.1              # [reserved]
    min_learning_rate: float = 1e-7           # [reserved]
    use_mixed_precision: bool = True          # [reserved]

    def validate(self):
        if self.lora_finetune_epochs <= 0:
            raise ConfigError("lora_finetune_epochs must be positive")
        if self.lora_finetune_lr <= 0:
            raise ConfigError("lora_finetune_lr must be positive")
        if self.batch_size < 1:
            raise ConfigError("batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ConfigError(
                "gradient_accumulation_steps must be >= 1")
        if self.weight_decay < 0:
            raise ConfigError("weight_decay must be >= 0")
        # warn about reserved fields at non-default values
        if self.lr_scheduler != "cosine_warmup":
            warnings.warn(
                f"lr_scheduler='{self.lr_scheduler}' is set but "
                f"not yet wired into AgentTrainer")


@dataclass
class BaselineConfig:
    """Ablation baseline runners for RQ1 comparison.

    IMPORTANT METHODOLOGICAL NOTE (SS4.2):
    These are NOT reimplementations of SmartAuditFlow or MANDO-LLM.
    They are ablation variants that test CrossGuard's architecture
    under conditions *analogous* to prior paradigms:

      run_smartauditflow  -> single_agent_llm ablation
      run_mando_llm       -> per_contract_no_cross ablation

    Field names match default.yaml for backward compatibility.
    """
    run_smartauditflow: bool = True    # single-agent LLM ablation
    run_mando_llm: bool = True         # per-contract no-cross ablation
    run_slither: bool = True
    run_mythril: bool = True
    slither_path: str = "slither"
    mythril_path: str = "myth"
    tool_timeout: int = 300            # seconds per invocation

    def validate(self):
        if self.tool_timeout <= 0:
            raise ConfigError("tool_timeout must be positive")


@dataclass
class DataConfig:
    """Dataset paths and splitting protocol (SS4.1).

    Three datasets:
      ESC      — 9 742 contracts, 70/15/15 (contract-level)
      SMS      — 514 880 functions, 80/20 × 5 runs (function-level)
      DAppSCAN — 682 DApps, 70/15/15 (DApp-level, PRIMARY)
    """
    active_datasets: List[str] = field(
        default_factory=lambda: ["esc", "sms", "dappscan"])

    # ── ESC ────────────────────────────────────────────────────────
    esc_data_dir: str = "data/dataset1_esc/raw"
    esc_train_split: float = 0.7
    esc_val_split: float = 0.15
    esc_test_split: float = 0.15

    # ── SMS ────────────────────────────────────────────────────────
    sms_data_dir: str = "data/dataset2_sms/raw"
    sms_train_split: float = 0.8
    sms_val_split: float = 0.0        # paper: no validation split
    sms_test_split: float = 0.2
    sms_num_runs: int = 5

    # ── DAppSCAN (primary) ─────────────────────────────────────────
    dappscan_source_dir: str = "data/dataset3_dappscan/source"
    dappscan_bytecode_dir: str = "data/dataset3_dappscan/bytecode"
    dappscan_train_split: float = 0.7
    dappscan_val_split: float = 0.15
    dappscan_test_split: float = 0.15
    dapp_level_split: bool = True      # prevent cross-file leakage

    # ── loader ─────────────────────────────────────────────────────
    num_workers: int = 4
    max_source_length: int = 8192

    def validate(self):
        if not self.dapp_level_split:
            warnings.warn(
                "dapp_level_split=False risks cross-file leakage "
                "on DAppSCAN (see SS4.1)")
        # Check split ratios sum to ~1.0
        for name, splits in [
            ("ESC", (self.esc_train_split,
                     self.esc_val_split,
                     self.esc_test_split)),
            ("DAppSCAN", (self.dappscan_train_split,
                          self.dappscan_val_split,
                          self.dappscan_test_split)),
        ]:
            total = sum(splits)
            if not math.isclose(total, 1.0, abs_tol=0.01):
                raise ConfigError(
                    f"{name} split ratios sum to {total:.3f}, "
                    f"expected ~1.0")
        sms_total = self.sms_train_split + self.sms_val_split + \
                    self.sms_test_split
        if not math.isclose(sms_total, 1.0, abs_tol=0.01):
            raise ConfigError(
                f"SMS split ratios sum to {sms_total:.3f}, "
                f"expected ~1.0")
        for ds in self.active_datasets:
            if ds not in ("esc", "sms", "dappscan"):
                raise ConfigError(f"Unknown dataset: '{ds}'")


@dataclass
class PathConfig:
    """Output directory paths.

    Paths are stored as *strings* internally so that the dataclass
    round-trips cleanly through JSON/YAML.  Use resolve() to obtain
    absolute Path objects when needed.
    """
    project_root: str = "."
    results_dir: str = "results"
    checkpoint_dir: str = "results/checkpoints"
    figures_dir: str = "results/figures"
    tables_dir: str = "results/tables"
    logs_dir: str = "results/logs"
    metrics_dir: str = "results/metrics"
    prompts_dir: str = "prompts"

    _DIR_FIELDS = (
        "results_dir", "checkpoint_dir", "figures_dir",
        "tables_dir", "logs_dir", "metrics_dir", "prompts_dir")

    def resolve(self, attr: str) -> Path:
        """Return the absolute Path for a directory field."""
        return Path(self.project_root).resolve() / getattr(self, attr)

    def create_directories(self):
        """Create all output directories."""
        for a in self._DIR_FIELDS:
            self.resolve(a).mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name: str):
        # Allow transparent access as Path objects for downstream code
        # that does Path(config.paths.figures_dir).
        # (Only triggered when normal attribute lookup fails, which
        # should not happen for declared fields.)
        raise AttributeError(
            f"PathConfig has no attribute '{name}'")


@dataclass
class ReproducibilityConfig:
    """Reproducibility settings (SS4.3)."""
    seed: int = 42
    deterministic: bool = True
    num_bootstrap_samples: int = 1000  # for confidence intervals
    num_permutation_tests: int = 10000 # for p-values

    def set_seed(self):
        """Set all random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def validate(self):
        if self.num_bootstrap_samples < 100:
            raise ConfigError(
                "num_bootstrap_samples should be >= 100 for "
                "reliable confidence intervals")


@dataclass
class ExperimentConfig:
    """Experiment orchestration settings."""
    experiment_name: str = "crossguard_tosem"
    run_baselines: bool = True        # run ablation baselines in RQ1

    # ── ablation variants (RQ3, SS5.4) ─────────────────────────────
    ablation_variants: List[str] = field(default_factory=lambda: [
        "no_cross_contract",
        "no_interaction_graph",
        "no_message_passing",
        "no_local_analysis",
        "no_llm",
        "single_agent",
        "no_synthesis",
    ])


@dataclass
class LoggingConfig:
    """Logging and output settings."""
    log_level: str = "info"
    log_interval: int = 10
    save_agent_messages: bool = True   # persist inter-agent messages


# ═══════════════════════════════════════════════════════════════════════
# Top-level Config
# ═══════════════════════════════════════════════════════════════════════

class Config:
    """Unified configuration for the CrossGuard framework.

    Typical usage::

        config = Config.load("configs/default.yaml")
        config.setup()      # validate, create dirs, set seeds
        # ... run experiments ...
        config.save("results/config_used.yaml")
    """

    # Sub-config sections and their YAML keys.
    _SECTIONS = (
        ("agent",           "agent",           AgentConfig),
        ("protocol",        "protocol",        ProtocolConfig),
        ("decomposer",      "decomposer",      DecomposerConfig),
        ("training",        "training",        TrainingConfig),
        ("baselines",       "baselines",       BaselineConfig),
        ("data",            "data",            DataConfig),
        ("paths",           "paths",           PathConfig),
        ("reproducibility", "reproducibility", ReproducibilityConfig),
        ("experiment",      "experiment",      ExperimentConfig),
        ("logging_cfg",     "logging",         LoggingConfig),
    )

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        d = cfg or {}
        for attr, yaml_key, cls in self._SECTIONS:
            raw = d.get(yaml_key, {})
            # Filter out keys that the dataclass does not accept,
            # so loading a YAML with extra keys does not crash.
            valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in raw.items()
                        if k in valid_fields}
            dropped = set(raw) - valid_fields
            if dropped:
                logger.debug(
                    f"Config section '{yaml_key}': ignoring unknown "
                    f"keys {dropped}")
            setattr(self, attr, cls(**filtered))
        self.creation_time = datetime.now().isoformat()

    # ── validation ─────────────────────────────────────────────────

    def validate(self):
        """Run validation on every sub-config that defines it."""
        for attr, _, _ in self._SECTIONS:
            obj = getattr(self, attr)
            if hasattr(obj, 'validate'):
                obj.validate()

    # ── setup ──────────────────────────────────────────────────────

    def setup(self):
        """Validate, create output directories, set seeds, log banner.

        Call this once at the start of an experiment.
        """
        self.validate()
        self.paths.create_directories()
        self.reproducibility.set_seed()
        self._log_banner()

    def _log_banner(self):
        """Print experiment configuration summary."""
        if torch.cuda.is_available():
            dev = (f"CUDA x{torch.cuda.device_count()} "
                   f"({torch.cuda.get_device_name(0)})")
        else:
            dev = "CPU"
        eff_batch = (self.training.batch_size
                     * self.training.gradient_accumulation_steps)
        lines = [
            "",
            "=" * 70,
            f"CROSSGUARD — {self.experiment.experiment_name}",
            "=" * 70,
            f"  LLM        : {self.agent.llm_model}",
            f"  QLoRA      : rank={self.agent.lora_rank}, "
            f"alpha={self.agent.lora_alpha}, "
            f"bits={self.agent.quantisation_bits}",
            f"  Phase 2    : LLM={self.agent.use_llm_cross_reasoning}, "
            f"T={self.protocol.num_reasoning_rounds}, "
            f"theta={self.protocol.convergence_threshold}",
            f"  Training   : {self.training.lora_finetune_epochs} epochs, "
            f"lr={self.training.lora_finetune_lr}, "
            f"eff_batch={eff_batch}",
            f"  Datasets   : {', '.join(self.data.active_datasets)}",
            f"  Device     : {dev}",
            f"  Seed       : {self.reproducibility.seed}",
            "=" * 70,
            "",
        ]
        banner = "\n".join(lines)
        logger.info(banner)
        # Also print to stdout for CLI usage
        print(banner)

    # ── serialisation ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON/YAML-safe).

        Path objects are converted to strings.  All sub-configs are
        included so that save + load round-trips faithfully.
        """
        result = {}
        for attr, yaml_key, _ in self._SECTIONS:
            d = asdict(getattr(self, attr))
            # Convert any Path objects to strings for serialisation
            for k, v in d.items():
                if isinstance(v, Path):
                    d[k] = str(v)
            result[yaml_key] = d
        return result

    def save(self, fp: Union[str, Path]):
        """Save configuration to YAML or JSON."""
        fp = Path(fp)
        fp.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with open(fp, "w") as f:
            if fp.suffix in (".yaml", ".yml"):
                yaml.dump(data, f, default_flow_style=False,
                          sort_keys=False)
            else:
                json.dump(data, f, indent=2, default=str)
        logger.info(f"Config saved: {fp}")

    @classmethod
    def load(cls, fp: Union[str, Path]) -> "Config":
        """Load configuration from YAML or JSON."""
        fp = Path(fp)
        with open(fp) as f:
            if fp.suffix in (".yaml", ".yml"):
                d = yaml.safe_load(f)
            else:
                d = json.load(f)
        logger.info(f"Config loaded: {fp}")
        return cls(d)


# ═══════════════════════════════════════════════════════════════════════
# Convenience constructors
# ═══════════════════════════════════════════════════════════════════════

def get_default_config() -> Config:
    """Return a Config with all default values."""
    return Config()


def create_ablation_configs(base: Config) -> Dict[str, Config]:
    """Create the seven ablation variants for RQ3 (SS5.4, Table 6).

    Each variant removes or degrades one architectural component while
    keeping all others unchanged.  The variants are:

    no_cross_contract     T=0, disables Phase 2 entirely
    no_interaction_graph  all six edge detectors off (random pairing)
    no_message_passing    T=1, no iterative refinement
    no_local_analysis     local token budget = 0 (cross-only)
    no_llm                MLP fallback, no LLM reasoning
    single_agent          one agent for entire DApp
    no_synthesis          score threshold = 0 (no filtering)
    """
    specs = [
        ("no_cross_contract", {
            "protocol": {
                "num_reasoning_rounds": 0,
            },
        }),
        ("no_interaction_graph", {
            # Disable ALL six edge types so agents have no structural
            # information about which contracts are coupled.
            "decomposer": {
                "detect_inheritance": False,
                "detect_external_calls": False,
                "detect_interface_deps": False,
                "detect_state_deps": False,
                "detect_event_deps": False,
                "detect_proxy_patterns": False,
            },
        }),
        ("no_message_passing", {
            "protocol": {
                "num_reasoning_rounds": 1,
                "use_iterative_refinement": False,
            },
        }),
        ("no_local_analysis", {
            "agent": {
                "local_analysis_max_tokens": 0,
            },
        }),
        ("no_llm", {
            "agent": {
                "use_fallback_mlp": True,
                "use_llm_cross_reasoning": False,
            },
        }),
        ("single_agent", {
            "agent": {
                "max_agents_per_dapp": 1,
            },
        }),
        ("no_synthesis", {
            "protocol": {
                "vulnerability_score_threshold": 0.0,
            },
        }),
    ]

    variants: Dict[str, Config] = {}
    for name, overrides in specs:
        # Deep-copy via serialisation round-trip
        c = Config(base.to_dict())
        c.experiment.experiment_name = f"ablation_{name}"
        for section_yaml_key, params in overrides.items():
            # Map YAML key to attribute name
            attr = section_yaml_key
            for a, yk, _ in Config._SECTIONS:
                if yk == section_yaml_key:
                    attr = a
                    break
            obj = getattr(c, attr)
            for k, v in params.items():
                setattr(obj, k, v)
        variants[name] = c
    return variants
