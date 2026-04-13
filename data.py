"""
Multi-dataset pipeline for CrossGuard.

Three datasets (SS4.1):
  ESC      9 742 contracts, 70/15/15 stratified, contract-level
  SMS      514 880 functions, 80/20 x 5 runs, function-level
  DAppSCAN 682 DApps, 70/15/15 stratified DApp-level (PRIMARY)

DAppSCAN is the PRIMARY evaluation target because it preserves the
multi-contract structure of real-world DApps.  ESC and SMS are wrapped
as single-contract DApps so they flow through the same pipeline.

Author : [Anonymous for double-blind review]
Target : ACM Transactions on Software Engineering and Methodology
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset, DataLoader

from .decomposer import CrossContractGroundTruth

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Core data structure
# ═══════════════════════════════════════════════════════════════════════

class DAppSample:
    """A single DApp sample, possibly containing multiple contracts.

    For ESC and SMS, each sample wraps a single Solidity source file.
    For DAppSCAN, a sample contains all source files belonging to one
    DApp project.

    Note: ``num_files`` counts the number of ``.sol`` files in the
    sample, NOT the number of parsed contract definitions.  A single
    file may define multiple contracts.  The actual contract count is
    determined by the DAppDecomposer at analysis time.
    """
    __slots__ = ("dapp_id", "files", "vulnerabilities", "dataset",
                 "metadata", "cross_contract_gt")

    def __init__(self, dapp_id: str = "", files: Optional[Dict[str, str]] = None,
                 vulnerabilities: Optional[List[Dict[str, Any]]] = None,
                 dataset: str = "",
                 metadata: Optional[Dict[str, Any]] = None,
                 cross_contract_gt: Optional[List[Dict[str, Any]]] = None):
        self.dapp_id = dapp_id
        self.files: Dict[str, str] = files or {}
        self.vulnerabilities: List[Dict[str, Any]] = vulnerabilities or []
        self.dataset = dataset
        self.metadata: Dict[str, Any] = metadata or {}
        self.cross_contract_gt: List[Dict[str, Any]] = cross_contract_gt or []

    @property
    def num_files(self) -> int:
        """Number of Solidity source files in this sample."""
        return len(self.files)

    @property
    def label(self) -> int:
        """Binary label: 1 = vulnerable, 0 = safe."""
        return 1 if self.vulnerabilities else 0

    @property
    def has_cross_contract_vuln(self) -> bool:
        """Whether the sample has cross-contract vulnerability indicators."""
        return len(self.cross_contract_gt) > 0

    @property
    def total_loc(self) -> int:
        """Total lines of code across all source files."""
        return sum(c.count("\n") + 1 for c in self.files.values())

    def __repr__(self) -> str:
        return (f"DAppSample({self.dapp_id!r}, "
                f"files={self.num_files}, "
                f"label={self.label}, "
                f"dataset={self.dataset!r})")


# ═══════════════════════════════════════════════════════════════════════
# Dataset loaders
# ═══════════════════════════════════════════════════════════════════════

class Dataset1_ESC:
    """ESC (Ethereum Smart Contracts) — Zhuang et al., IJCAI 2020.

    9 742 contracts, each labelled with one of four classes:
    reentrancy (1), timestamp dependence (2), integer overflow (3),
    or safe (0).

    Expected directory layout::

        data_dir/
            graph_index.txt     # lines: <contract_id> <label_int>
            source_code/
                <contract_id>.sol
    """

    LABEL_MAP = {
        0: "safe",
        1: "reentrancy",
        2: "timestamp",
        3: "overflow",
    }
    SWC_MAP = {
        1: 107,   # reentrancy
        2: 116,   # timestamp dependence
        3: 101,   # integer overflow
    }

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> List[DAppSample]:
        logger.info("Loading Dataset 1 (ESC) ...")
        samples: List[DAppSample] = []
        idx_file = self.data_dir / "graph_index.txt"
        src_dir = self.data_dir / "source_code"

        if not idx_file.exists():
            logger.warning(
                f"ESC index file not found: {idx_file}. "
                f"Returning empty dataset.")
            return samples

        with open(idx_file) as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                cid = parts[0]
                try:
                    lbl = int(parts[1])
                except ValueError:
                    logger.debug(
                        f"ESC line {line_num}: cannot parse label "
                        f"'{parts[1]}', skipping")
                    continue

                code = ""
                fp = src_dir / f"{cid}.sol"
                if fp.exists():
                    try:
                        code = fp.read_text(errors="replace")
                    except OSError as exc:
                        logger.debug(f"ESC: cannot read {fp}: {exc}")

                if lbl == 0:
                    vulns: List[Dict[str, Any]] = []
                else:
                    vulns = [{
                        "type": self.LABEL_MAP.get(lbl, "unknown"),
                        "swc_id": self.SWC_MAP.get(lbl, -1),
                    }]

                samples.append(DAppSample(
                    dapp_id=cid,
                    files={f"{cid}.sol": code},
                    vulnerabilities=vulns,
                    dataset="esc"))

        # Log class distribution
        dist = Counter(s.label for s in samples)
        logger.info(
            f"  ESC: {len(samples)} contracts "
            f"(vuln={dist.get(1, 0)}, safe={dist.get(0, 0)})")
        return samples


class Dataset2_SMS:
    """SMS dataset — Qian et al., WWW 2023.

    514 880 functions from 42 910 contracts, labelled for four
    vulnerability types.  Detection is formulated as four independent
    binary classification tasks at the function level.

    Expected directory layout::

        data_dir/
            reentrancy/
                vulnerable/ or files with "vuln" in path
                safe/ or files without "vuln" in path
            timestamp/
            overflow/
            delegatecall/
    """

    TYPES = ["reentrancy", "timestamp", "overflow", "delegatecall"]
    SWC = {
        "reentrancy": 107,
        "timestamp": 116,
        "overflow": 101,
        "delegatecall": 112,
    }

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> List[DAppSample]:
        logger.info("Loading Dataset 2 (SMS) ...")
        samples: List[DAppSample] = []

        for vt in self.TYPES:
            # Try multiple possible directory layouts
            candidates = [
                self.data_dir / vt,
                self.data_dir / "source_code" / vt,
            ]
            vdir = None
            for c in candidates:
                if c.is_dir():
                    vdir = c
                    break
            if vdir is None:
                logger.debug(
                    f"SMS: no directory found for '{vt}', skipping")
                continue

            n_vuln, n_safe = 0, 0
            for fp in sorted(vdir.rglob("*.sol")):
                try:
                    code = fp.read_text(errors="replace")
                except OSError as exc:
                    logger.debug(f"SMS: cannot read {fp}: {exc}")
                    continue

                # Label: "vuln" substring in path indicates vulnerable
                is_vuln = "vuln" in str(fp).lower()
                if is_vuln:
                    vulns: List[Dict[str, Any]] = [{
                        "type": vt,
                        "swc_id": self.SWC.get(vt, -1),
                    }]
                    n_vuln += 1
                else:
                    vulns = []
                    n_safe += 1

                samples.append(DAppSample(
                    dapp_id=fp.stem,
                    files={fp.name: code},
                    vulnerabilities=vulns,
                    dataset="sms",
                    metadata={"vuln_type": vt}))

            logger.info(
                f"  SMS/{vt}: {n_vuln + n_safe} samples "
                f"(vuln={n_vuln}, safe={n_safe})")

        logger.info(f"  SMS total: {len(samples)} samples")
        return samples


class Dataset3_DAppSCAN:
    """DAppSCAN — Zheng et al., IEEE TSE 2024.  PRIMARY dataset.

    682 DApp projects with 39 904 Solidity source files and 1 618
    annotated SWC weaknesses.  Preserves multi-contract structure.

    Expected directory layout::

        source_dir/
            <project_name>/
                *.sol files (possibly in subdirectories)
                *.json weakness reports

    The loader groups all ``.sol`` files under each top-level project
    directory into a single DAppSample.  Subdirectories within a
    project (e.g., ``contracts/``, ``interfaces/``) are traversed
    recursively.
    """

    def __init__(self, source_dir: str,
                 bytecode_dir: Optional[str] = None):
        self.source_dir = Path(source_dir)

    def load(self) -> List[DAppSample]:
        logger.info("Loading Dataset 3 (DAppSCAN) — multi-contract ...")
        dapps: Dict[str, DAppSample] = {}

        if not self.source_dir.exists():
            logger.warning(
                f"DAppSCAN source directory not found: "
                f"{self.source_dir}. Returning empty dataset.")
            return []

        # ── group .sol files by top-level project directory ────────
        for sol in sorted(self.source_dir.rglob("*.sol")):
            dapp_name = self._resolve_dapp_name(sol)
            if dapp_name is None:
                continue

            if dapp_name not in dapps:
                dapps[dapp_name] = DAppSample(
                    dapp_id=dapp_name,
                    files={},
                    vulnerabilities=[],
                    dataset="dappscan",
                    metadata={"dapp_dir": str(sol.parent)})

            # Use path relative to project root as file key to avoid
            # collisions from files with the same name in subdirs
            try:
                rel = sol.relative_to(self.source_dir / dapp_name)
                file_key = str(rel)
            except ValueError:
                file_key = sol.name

            try:
                dapps[dapp_name].files[file_key] = \
                    sol.read_text(errors="replace")
            except OSError as exc:
                logger.debug(f"DAppSCAN: cannot read {sol}: {exc}")

        # ── load vulnerability annotations from JSON reports ──────
        for jf in sorted(self.source_dir.rglob("*.json")):
            try:
                with open(jf) as f:
                    report = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug(
                    f"DAppSCAN: cannot parse {jf}: {exc}")
                continue

            dapp_name = report.get("dapp", None)
            if dapp_name is None:
                # Try to infer from directory structure
                dapp_name = self._resolve_dapp_name(jf)
            if dapp_name is None or dapp_name not in dapps:
                continue

            weaknesses = report.get(
                "SWCs", report.get("weaknesses", []))
            for swc in weaknesses:
                cat = swc.get("category", "")
                m = re.search(r'SWC-(\d+)', str(cat))
                dapps[dapp_name].vulnerabilities.append({
                    "type": cat,
                    "swc_id": int(m.group(1)) if m else -1,
                    "function": swc.get("function", ""),
                    "file": swc.get("filePath", ""),
                    "line": swc.get("lineNumber", ""),
                })

        # ── extract cross-contract ground truth (SS4.1) ───────────
        for dapp in dapps.values():
            dapp.cross_contract_gt = CrossContractGroundTruth.extract(
                dapp.vulnerabilities, dapp.files)

        samples = list(dapps.values())

        # Log corpus statistics
        n_multi = sum(1 for s in samples if s.num_files > 1)
        n_cross = sum(1 for s in samples if s.has_cross_contract_vuln)
        n_vuln = sum(1 for s in samples if s.label == 1)
        total_files = sum(s.num_files for s in samples)
        avg_files = total_files / max(1, len(samples))
        logger.info(
            f"  DAppSCAN: {len(samples)} DApps, "
            f"{total_files} files (avg {avg_files:.1f}/DApp)")
        logger.info(
            f"  DAppSCAN: {n_multi} multi-file, "
            f"{n_vuln} vulnerable, "
            f"{n_cross} with cross-contract indicators")
        return samples

    def _resolve_dapp_name(self, filepath: Path) -> Optional[str]:
        """Resolve a file path to its top-level DApp project name.

        Given source_dir/ProjectX/contracts/Token.sol, returns
        "ProjectX".  Returns None if the file is directly in
        source_dir (no project subdirectory).
        """
        try:
            rel = filepath.relative_to(self.source_dir)
        except ValueError:
            return None
        parts = rel.parts
        if len(parts) < 2:
            # File is directly in source_dir, no project grouping
            return None
        return parts[0]


# ═══════════════════════════════════════════════════════════════════════
# PyTorch Dataset wrapper
# ═══════════════════════════════════════════════════════════════════════

class DAppDataset(Dataset):
    """Wraps a list of DAppSample into a PyTorch Dataset.

    Each item is returned as a plain dict so that it can be consumed
    by the CrossGuard pipeline without tensor conversion.
    """

    def __init__(self, samples: List[DAppSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "dapp_id": s.dapp_id,
            "files": s.files,
            "vulnerabilities": s.vulnerabilities,
            "label": s.label,
            "num_contracts": s.num_files,  # backward compat key name
            "num_files": s.num_files,
            "dataset": s.dataset,
            "total_loc": s.total_loc,
            "has_cross_contract_vuln": s.has_cross_contract_vuln,
            "cross_contract_gt": s.cross_contract_gt,
        }


def dapp_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identity collate: returns list of sample dicts unchanged.

    Standard torch collation would fail on variable-length dicts of
    source code strings.  The pipeline iterates samples individually.
    """
    return batch


# ═══════════════════════════════════════════════════════════════════════
# Stratified splitting
# ═══════════════════════════════════════════════════════════════════════

class DAppSplitter:
    """Stratified train/val/test splitting for DApp samples.

    Paper (SS4.1):
      ESC:      "stratified random sampling at the contract level"
      DAppSCAN: "stratified by the presence or absence of annotated
                 vulnerabilities to maintain class balance"
      SMS:      "80/20 train/test split" (no validation set)

    Uses sklearn's train_test_split with stratification on the binary
    vulnerability label to maintain class balance across partitions.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def split(self, samples: List[DAppSample],
              train_ratio: float,
              val_ratio: float,
              test_ratio: float,
              ) -> Tuple[List[int], Optional[List[int]], List[int]]:
        """Split sample indices into train / val / test.

        Returns (train_indices, val_indices, test_indices).
        val_indices is None when val_ratio is 0 (e.g., SMS).

        All splits are stratified on ``sample.label`` to maintain
        class balance.
        """
        n = len(samples)
        if n == 0:
            return [], None, []

        indices = np.arange(n)
        labels = np.array([s.label for s in samples])

        # ── split off test set ─────────────────────────────────────
        test_size = max(1, int(round(n * test_ratio)))

        # Guard: stratification requires at least 2 members per class
        # so that both train and test can receive at least 1.
        # Fall back to non-stratified when this is not satisfied.
        class_counts = Counter(labels.tolist())
        can_stratify = (len(class_counts) >= 2
                        and min(class_counts.values()) >= 2)
        stratify = labels if can_stratify else None

        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=self.seed,
            stratify=stratify)

        # ── split off validation set (if requested) ────────────────
        if val_ratio > 0 and len(train_val_idx) > 1:
            # val_ratio is relative to the full dataset; rescale to
            # the remaining train+val portion.
            remaining = len(train_val_idx) / n
            val_frac = val_ratio / remaining
            val_frac = min(val_frac, 0.5)  # safety cap
            val_size = max(1, int(round(len(train_val_idx) * val_frac)))

            tv_labels = labels[train_val_idx]
            tv_counts = Counter(tv_labels.tolist())
            tv_can_stratify = (len(tv_counts) >= 2
                               and min(tv_counts.values()) >= 2)
            tv_stratify = tv_labels if tv_can_stratify else None

            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_size,
                random_state=self.seed,
                stratify=tv_stratify)

            return (train_idx.tolist(),
                    val_idx.tolist(),
                    test_idx.tolist())
        else:
            # No validation split (e.g., SMS 80/20)
            return (train_val_idx.tolist(),
                    None,
                    test_idx.tolist())

    def cv_folds(self, samples: List[DAppSample],
                 n_folds: int = 5,
                 ) -> List[Tuple[List[int], List[int]]]:
        """Stratified k-fold cross-validation splits.

        Used for SMS evaluation: "80/20 train/test split with five
        independent runs, results averaged across runs" (SS4.1).
        """
        labels = np.array([s.label for s in samples])
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True,
            random_state=self.seed)
        return [(train.tolist(), test.tolist())
                for train, test in skf.split(
                    np.arange(len(samples)), labels)]


# ═══════════════════════════════════════════════════════════════════════
# DataLoader factory
# ═══════════════════════════════════════════════════════════════════════

def create_dataloaders(
        config,
        dataset_name: Optional[str] = None,
) -> Dict[str, DataLoader]:
    """Create train / val / test DataLoaders for one dataset.

    Returns a dict with keys "train", "test", and optionally "val".
    The "val" key is absent when the dataset does not use a validation
    split (e.g., SMS with 80/20 train/test).

    Parameters
    ----------
    config : Config
        Full CrossGuard configuration.
    dataset_name : str, optional
        Dataset to load ("esc", "sms", "dappscan").  Defaults to
        the first entry in ``config.data.active_datasets``.
    """
    target = dataset_name or config.data.active_datasets[0]
    splitter = DAppSplitter(config.reproducibility.seed)

    if target == "esc":
        samples = Dataset1_ESC(config.data.esc_data_dir).load()
        train_idx, val_idx, test_idx = splitter.split(
            samples,
            train_ratio=config.data.esc_train_split,
            val_ratio=config.data.esc_val_split,
            test_ratio=config.data.esc_test_split)

    elif target == "sms":
        samples = Dataset2_SMS(config.data.sms_data_dir).load()
        train_idx, val_idx, test_idx = splitter.split(
            samples,
            train_ratio=config.data.sms_train_split,
            val_ratio=config.data.sms_val_split,
            test_ratio=config.data.sms_test_split)

    elif target == "dappscan":
        samples = Dataset3_DAppSCAN(
            config.data.dappscan_source_dir,
            config.data.dappscan_bytecode_dir).load()
        train_idx, val_idx, test_idx = splitter.split(
            samples,
            train_ratio=config.data.dappscan_train_split,
            val_ratio=config.data.dappscan_val_split,
            test_ratio=config.data.dappscan_test_split)

    else:
        raise ValueError(f"Unknown dataset: {target}")

    # ── build DataLoaders ──────────────────────────────────────────
    loaders: Dict[str, DataLoader] = {}
    splits: List[Tuple[str, Optional[List[int]]]] = [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ]

    for name, idx in splits:
        if idx is None:
            continue                     # no val split for this dataset
        ds = DAppDataset([samples[i] for i in idx])
        loaders[name] = DataLoader(
            ds,
            batch_size=config.training.batch_size,
            shuffle=(name == "train"),
            num_workers=0,               # DApp samples are not picklable
            collate_fn=dapp_collate)

    # ── log split statistics ───────────────────────────────────────
    for name, loader in loaders.items():
        n = len(loader.dataset)
        labels = [loader.dataset[i]["label"] for i in range(n)]
        n_pos = sum(labels)
        logger.info(
            f"  {target}/{name}: {n} samples "
            f"(pos={n_pos}, neg={n - n_pos}, "
            f"pos_rate={n_pos / max(1, n):.3f})")

    return loaders
