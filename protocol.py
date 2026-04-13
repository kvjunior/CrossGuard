"""
Cross-contract reasoning protocol for CrossGuard.

Implements Algorithm 1 from the paper (SS3.3):
  Round 0:    initial exchange of local summaries along edges
  Rounds 1-T: iterative LLM-driven cross-contract reasoning
  Multi-hop:  attack-path detection for chains >= 3 contracts

Also contains:
  ConvergenceTracker      message-similarity tracking for RQ5
  CrossContractPatterns   eight vulnerability pattern families
  VulnerabilitySynthesiser Phase 3 scoring and ranking (SS3.4)
  CaseStudyReport        structured output for RQ6

Author : [Anonymous for double-blind review]
Target : ACM Transactions on Software Engineering and Methodology
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .decomposer import InteractionGraph
from .agents import AgentPool, AgentSummary, LLMModule

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CrossContractMessage:
    """A single message exchanged between agents during Phase 2."""
    from_agent: str
    to_agent: str
    round_num: int
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    content_hash: str = ""

    def compute_hash(self) -> str:
        """Content fingerprint for convergence tracking (SS3.3).

        Constructed by sorting the vulnerability types reported in the
        message and appending the risk-delta value.  Used by
        ConvergenceTracker to compare messages across rounds.
        """
        vulns = self.content.get("cross_vulns_found", [])
        types = sorted(
            v.get("type", "")
            for v in vulns
            if isinstance(v, dict))
        risk_delta = self.content.get("risk_delta", 0.0)
        self.content_hash = (
            "|".join(types) + f"|{risk_delta:.2f}")
        return self.content_hash


@dataclass
class CrossContractVulnerability:
    """A cross-contract vulnerability finding from Phase 2.

    Consumed by VulnerabilitySynthesiser in Phase 3 for scoring
    and ranking.
    """
    vuln_type: str
    attack_path: List[str]
    confidence: float
    severity: str
    description: str = ""
    evidence: List[str] = field(default_factory=list)
    swc_id: int = -1
    is_cross_contract: bool = True
    source: str = ""            # "llm" or "pattern"

    @property
    def path_length(self) -> int:
        return len(self.attack_path)

    def score(self) -> float:
        """Compute composite score per Equation (1) in SS3.4."""
        sev_w = _SEVERITY_WEIGHTS.get(self.severity, 0.5)
        path_pen = 1.0 / (1.0 + 0.1 * (self.path_length - 1))
        return self.confidence * sev_w * path_pen


# ═══════════════════════════════════════════════════════════════════════
# Convergence tracker
# ═══════════════════════════════════════════════════════════════════════

class ConvergenceTracker:
    """Track message similarity across rounds (SS3.3, RQ5).

    Computes similarity between consecutive rounds' message
    fingerprints using either Jaccard (over type sets) or cosine
    (over type-frequency vectors).

    Paper: "Message fingerprints are compared between consecutive
    rounds using either Jaccard similarity or cosine similarity."
    """

    def __init__(self, method: str = "cosine"):
        if method not in ("cosine", "jaccard"):
            raise ValueError(
                f"similarity_method must be 'cosine' or 'jaccard', "
                f"got '{method}'")
        self.method = method
        self.round_hashes: List[Dict[Tuple[str, str], str]] = []
        self.similarities: List[float] = []

    def record_round(
            self,
            messages: Dict[Tuple[str, str], CrossContractMessage]):
        """Record all messages from one round for convergence tracking.

        Computes and stores the similarity with the previous round.
        """
        hashes: Dict[Tuple[str, str], str] = {}
        for key, msg in messages.items():
            msg.compute_hash()
            hashes[key] = msg.content_hash
        self.round_hashes.append(hashes)

        if len(self.round_hashes) >= 2:
            sim = self._compute_similarity(
                self.round_hashes[-2],
                self.round_hashes[-1])
            self.similarities.append(sim)

    def is_converged(self, threshold: float) -> bool:
        """Check if messages have stabilised.

        Paper (SS3.3): "If the similarity between round t and
        round t-1 exceeds 1 - theta, the protocol terminates early."
        """
        if not self.similarities:
            return False
        return self.similarities[-1] >= (1.0 - threshold)

    def get_convergence_curve(self) -> List[float]:
        """Return per-round similarity values for RQ5 plotting."""
        return list(self.similarities)

    def _compute_similarity(
            self,
            prev: Dict[Tuple[str, str], str],
            curr: Dict[Tuple[str, str], str]) -> float:
        """Compute similarity between two rounds' message sets."""
        if self.method == "jaccard":
            return self._jaccard(prev, curr)
        return self._cosine(prev, curr)

    @staticmethod
    def _jaccard(prev: Dict, curr: Dict) -> float:
        """Jaccard similarity over message content hash sets."""
        prev_set = set(prev.values())
        curr_set = set(curr.values())
        if not prev_set and not curr_set:
            return 1.0
        intersection = len(prev_set & curr_set)
        union = len(prev_set | curr_set)
        return intersection / max(1, union)

    @staticmethod
    def _cosine(prev: Dict, curr: Dict) -> float:
        """Cosine similarity over vulnerability-type frequency vectors.

        Each message hash has the form "type1|type2|...|risk_delta".
        We extract the types (everything before the last |) and count
        their frequencies across all messages in the round.
        """
        def _extract_types(hashes: Dict) -> Counter:
            counts: Counter = Counter()
            for h in hashes.values():
                if not h:
                    continue
                parts = h.rsplit("|", 1)
                type_str = parts[0] if len(parts) > 1 else h
                for t in type_str.split("|"):
                    if t:
                        counts[t] += 1
            return counts

        prev_types = _extract_types(prev)
        curr_types = _extract_types(curr)
        all_types = set(prev_types) | set(curr_types)

        if not all_types:
            return 1.0

        dot = sum(prev_types.get(t, 0) * curr_types.get(t, 0)
                  for t in all_types)
        mag_a = sum(v ** 2 for v in prev_types.values()) ** 0.5
        mag_b = sum(v ** 2 for v in curr_types.values()) ** 0.5
        denom = mag_a * mag_b
        if denom < 1e-12:
            return 0.0
        return dot / denom


# ═══════════════════════════════════════════════════════════════════════
# Cross-contract vulnerability patterns
# ═══════════════════════════════════════════════════════════════════════

class CrossContractPatterns:
    """Eight cross-contract vulnerability pattern families (SS3.3).

    Each pattern describes a mechanism through which the interaction
    between two contracts can be exploited.  Severity and SWC-ID
    metadata are used by the VulnerabilitySynthesiser in Phase 3.

    Paper: "The cross-contract reasoning prompt directs the LLM
    toward eight vulnerability pattern families."
    """

    PATTERNS: Dict[str, Dict[str, Any]] = {
        "cross_reentrancy": {
            "severity": "critical",
            "swc_id": 107,
            "description": (
                "A calls B, B calls back to A before "
                "state update"),
        },
        "approval_abuse": {
            "severity": "high",
            "swc_id": 104,
            "description": (
                "Token approval in A exploited via B's "
                "transferFrom"),
        },
        "oracle_manipulation": {
            "severity": "critical",
            "swc_id": 114,
            "description": (
                "Price oracle in A manipulated to exploit B"),
        },
        "proxy_storage_collision": {
            "severity": "critical",
            "swc_id": 112,
            "description": (
                "Proxy and implementation have incompatible "
                "storage layout"),
        },
        "flash_loan_attack": {
            "severity": "critical",
            "swc_id": 114,
            "description": (
                "Flash loan from A used to exploit price in B"),
        },
        "shared_state_corruption": {
            "severity": "high",
            "swc_id": 107,
            "description": (
                "Multiple contracts modify shared state "
                "without mutual exclusion"),
        },
        "callback_chain": {
            "severity": "high",
            "swc_id": 107,
            "description": (
                "Callback chain across contracts causes "
                "unexpected state mutations"),
        },
        "privilege_escalation": {
            "severity": "high",
            "swc_id": 105,
            "description": (
                "Inherited permissions allow unintended "
                "cross-contract access"),
        },
    }

    @classmethod
    def get(cls, name: str) -> Dict[str, Any]:
        """Get pattern metadata by name."""
        return cls.PATTERNS.get(name, {})

    @classmethod
    def all_names(cls) -> List[str]:
        """List all pattern names."""
        return list(cls.PATTERNS.keys())


# ═══════════════════════════════════════════════════════════════════════
# Cross-contract reasoning protocol (Algorithm 1)
# ═══════════════════════════════════════════════════════════════════════

class CrossContractReasoningProtocol:
    """Structured protocol for inter-agent cross-contract reasoning.

    Implements Algorithm 1 from the paper (SS3.3):

      Round 0:      each agent broadcasts its AgentSummary to every
                    neighbour in the interaction graph.
      Rounds 1..T: each agent performs LLM-driven cross-contract
                    reasoning for every edge it participates in.
      Convergence:  messages are fingerprinted and compared across
                    rounds; early termination when similarity
                    exceeds 1 - theta.
      Multi-hop:    depth-first enumeration of call chains spanning
                    >= 3 contracts with risk thresholds at endpoints.
    """

    def __init__(self, config,
                 graph: InteractionGraph,
                 pool: AgentPool):
        self.config = config
        self.graph = graph
        self.pool = pool
        self.message_history: List[CrossContractMessage] = []
        self.all_cross_vulns: List[CrossContractVulnerability] = []

        # Convergence tracker
        if config.protocol.track_message_similarity:
            method = config.protocol.similarity_method
        else:
            method = "cosine"
        self.convergence = ConvergenceTracker(method)

    def run(self, summaries: Dict[str, AgentSummary],
            ) -> List[CrossContractVulnerability]:
        """Execute the full cross-contract reasoning protocol.

        Parameters
        ----------
        summaries : dict
            Per-contract AgentSummary from Phase 1.

        Returns
        -------
        list of CrossContractVulnerability
            Deduplicated cross-contract findings.
        """
        T = self.config.protocol.num_reasoning_rounds
        if T <= 0:
            logger.info("Cross-contract reasoning disabled (T=0)")
            return []

        n_edges = self.graph.num_interactions
        logger.info(
            f"Protocol: T={T}, {n_edges} edges, "
            f"theta={self.config.protocol.convergence_threshold}")

        # ── Round 0: initial exchange ──────────────────────────────
        prev = self._initial_exchange(summaries)
        self.convergence.record_round(prev)

        # ── Rounds 1..T: cross-contract reasoning ─────────────────
        for t in range(1, T + 1):
            curr = self._reasoning_round(t, summaries)
            self.convergence.record_round(curr)

            if (self.config.protocol.use_iterative_refinement
                    and self.convergence.is_converged(
                        self.config.protocol
                        .convergence_threshold)):
                logger.info(
                    f"  Converged at round {t} "
                    f"(sim={self.convergence.similarities[-1]:.3f})")
                break

        # ── Collect and deduplicate ────────────────────────────────
        self._collect_findings()
        self._detect_multihop()

        curve = self.convergence.get_convergence_curve()
        logger.info(
            f"Protocol complete: "
            f"{len(self.all_cross_vulns)} cross-contract findings "
            f"(convergence: {[f'{s:.3f}' for s in curve]})")
        return self.all_cross_vulns

    def _initial_exchange(
            self,
            summaries: Dict[str, AgentSummary],
            ) -> Dict[Tuple[str, str], CrossContractMessage]:
        """Round 0: agents broadcast local summaries to neighbours.

        Paper (SS3.3): "In round 0 (initial exchange), each agent
        broadcasts its AgentSummary to every neighbour in the
        interaction graph.  The broadcast is bidirectional."
        """
        messages: Dict[Tuple[str, str], CrossContractMessage] = {}

        for edge in self.graph.interactions:
            for src, tgt in [(edge.source, edge.target),
                             (edge.target, edge.source)]:
                if src not in summaries:
                    continue
                agent = self.pool.get_agent(tgt)
                if agent is None:
                    continue

                summary = summaries[src]
                msg = CrossContractMessage(
                    from_agent=src,
                    to_agent=tgt,
                    round_num=0,
                    content={
                        "local_summary": summary.to_message(),
                        "interface_exposures": [
                            {"name": ie.function_name,
                             "payable": ie.is_payable,
                             "access_ctrl": ie.has_access_control}
                            for ie in summary.interface_exposures[:8]
                        ],
                        "cross_vulns_found": [],
                        "risk_score": summary.risk_score,
                    },
                    timestamp=time.time())

                messages[(src, tgt)] = msg
                self.message_history.append(msg)
                agent.received_messages.append(
                    {"from": src, **msg.content})

        return messages

    def _reasoning_round(
            self,
            round_num: int,
            summaries: Dict[str, AgentSummary],
            ) -> Dict[Tuple[str, str], CrossContractMessage]:
        """Rounds 1..T: LLM-driven cross-contract reasoning.

        Paper (SS3.3): "each agent performs LLM-driven cross-contract
        reasoning for every edge it participates in."
        """
        messages: Dict[Tuple[str, str], CrossContractMessage] = {}

        for edge in self.graph.interactions:
            for src, tgt in [(edge.source, edge.target),
                             (edge.target, edge.source)]:
                agent = self.pool.get_agent(src)
                if agent is None or tgt not in summaries:
                    continue

                cross_vulns = agent.process_cross_contract(
                    summaries[tgt], edge)

                msg = CrossContractMessage(
                    from_agent=src,
                    to_agent=tgt,
                    round_num=round_num,
                    content={
                        "cross_vulns_found": cross_vulns,
                        "risk_delta": sum(
                            v.get("confidence", 0)
                            for v in cross_vulns),
                        "round": round_num,
                    },
                    timestamp=time.time())

                messages[(src, tgt)] = msg
                self.message_history.append(msg)

        return messages

    def _collect_findings(self):
        """Collect and deduplicate cross-contract findings.

        Paper (SS3.4): "two findings are considered duplicates if
        they share the same vulnerability type and the same set of
        involved contracts (regardless of ordering)."
        """
        seen: Set[Tuple] = set()

        for agent in self.pool.agents.values():
            for f in agent.cross_contract_findings:
                # Dedup key: sorted contract set + vulnerability type
                path = f.get("path", [])
                path_key = tuple(sorted(path))
                dedup_key = (path_key, f.get("type", ""))
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                pat = CrossContractPatterns.get(
                    f.get("type", ""))
                self.all_cross_vulns.append(
                    CrossContractVulnerability(
                        vuln_type=f.get("type", "unknown"),
                        attack_path=path,
                        confidence=f.get("confidence", 0.5),
                        severity=pat.get("severity", "medium"),
                        description=f.get("description", ""),
                        swc_id=f.get(
                            "swc_id",
                            pat.get("swc_id", -1)),
                        source=f.get("source", "pattern")))

    def _detect_multihop(self):
        """Detect multi-hop attack paths spanning >= 3 contracts.

        Paper (SS3.3): "For each path with k >= 3, the detector
        checks whether both the start agent and the end agent have
        risk scores exceeding 0.3."
        """
        # Collect existing cross-contract finding paths to avoid
        # emitting multihop findings that duplicate pairwise ones
        existing_paths: Set[Tuple[str, ...]] = set()
        for cv in self.all_cross_vulns:
            existing_paths.add(tuple(cv.attack_path))

        max_len = self.config.protocol.max_path_length
        for chain in self.graph.get_call_chains(max_len):
            if len(chain) < 3:
                continue

            # Skip if this exact path already exists
            chain_tuple = tuple(chain)
            if chain_tuple in existing_paths:
                continue

            start_agent = self.pool.get_agent(chain[0])
            end_agent = self.pool.get_agent(chain[-1])
            if start_agent is None or end_agent is None:
                continue

            start_risk = (start_agent.summary.risk_score
                          if start_agent.summary else 0.0)
            end_risk = (end_agent.summary.risk_score
                        if end_agent.summary else 0.0)

            if start_risk > 0.3 and end_risk > 0.3:
                avg_risk = 0.5 * (start_risk + end_risk)
                self.all_cross_vulns.append(
                    CrossContractVulnerability(
                        vuln_type="multihop_attack_path",
                        attack_path=chain,
                        confidence=avg_risk,
                        severity="high",
                        description=(
                            f"Multi-hop attack path: "
                            f"{' -> '.join(chain)}"),
                        source="pattern"))

    def get_convergence_data(self) -> Dict[str, Any]:
        """Return convergence statistics for RQ5 reporting."""
        return {
            "curve": self.convergence.get_convergence_curve(),
            "n_rounds_used": len(self.convergence.round_hashes),
            "final_similarity": (
                self.convergence.similarities[-1]
                if self.convergence.similarities else 0.0),
            "n_messages_total": len(self.message_history),
        }


# ═══════════════════════════════════════════════════════════════════════
# Vulnerability synthesiser (Phase 3)
# ═══════════════════════════════════════════════════════════════════════

class VulnerabilitySynthesiser:
    """Phase 3: aggregate, score, filter, and rank findings (SS3.4).

    Scoring function (Equation 1):
      score(r) = conf(r) * w_sev(r) * pen_path(r)

    where:
      conf    confidence in [0, 1] from LLM or pattern matcher
      w_sev   severity weight: critical=1.0, high=0.8, medium=0.5,
              low=0.3
      pen_path  path-length penalty: 1 / (1 + 0.1 * (path_len - 1))

    Findings below the vulnerability score threshold (default 0.5)
    are discarded.  The remaining findings are deduplicated and
    sorted by score in descending order.
    """

    def __init__(self, config,
                 llm: Optional[LLMModule] = None):
        self.config = config
        self.llm = llm

    def synthesise(
            self,
            local_summaries: Dict[str, AgentSummary],
            cross_vulns: List[CrossContractVulnerability],
            ) -> List[Dict[str, Any]]:
        """Produce the final ranked vulnerability report.

        Returns a list of finding dicts, each containing:
          type, contracts, path_length, confidence, severity, score,
          is_cross_contract, description, swc_id, source
        """
        all_findings: List[Dict[str, Any]] = []
        threshold = self.config.protocol.vulnerability_score_threshold

        # ── local findings ─────────────────────────────────────────
        for cname, summary in local_summaries.items():
            for lf in summary.local_findings:
                severity = _infer_local_severity(
                    lf.vulnerability_type)
                sev_w = _SEVERITY_WEIGHTS.get(severity, 0.5)
                score = lf.confidence * sev_w
                # Apply threshold to ALL findings (SS3.4)
                if score < threshold:
                    continue
                all_findings.append({
                    "type": lf.vulnerability_type,
                    "contracts": [cname],
                    "path_length": 1,
                    "confidence": lf.confidence,
                    "severity": severity,
                    "score": score,
                    "is_cross_contract": False,
                    "description": lf.description,
                    "swc_id": lf.swc_id,
                    # Preserve original source ("llm" or "pattern")
                    # so metrics can compute the 2x2 breakdown in
                    # Table IX (LLM/pattern x local/cross)
                    "source": lf.source,
                })

        # ── cross-contract findings ────────────────────────────────
        for cv in cross_vulns:
            path_pen = 1.0 / (
                1.0 + 0.1 * (cv.path_length - 1))
            sev_w = _SEVERITY_WEIGHTS.get(cv.severity, 0.5)
            score = cv.confidence * sev_w * path_pen
            if score < threshold:
                continue
            all_findings.append({
                "type": cv.vuln_type,
                "contracts": cv.attack_path,
                "path_length": cv.path_length,
                "confidence": cv.confidence,
                "severity": cv.severity,
                "score": score,
                "is_cross_contract": True,
                "description": cv.description,
                "evidence": cv.evidence,
                "swc_id": cv.swc_id,
                "source": cv.source,
            })

        # ── sort by score descending ───────────────────────────────
        all_findings.sort(key=lambda f: f["score"], reverse=True)
        return all_findings


# ═══════════════════════════════════════════════════════════════════════
# Case study report
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CaseStudyReport:
    """Structured case study output for RQ6 (SS5.6).

    Used by CrossGuardEvaluator.generate_case_studies() to produce
    both JSON output and LaTeX fragments for the paper.
    """
    dapp_id: str
    num_contracts: int
    graph_summary: str
    interaction_types: Dict[str, int] = field(default_factory=dict)
    local_findings_per_contract: Dict[str, int] = field(
        default_factory=dict)
    cross_contract_findings: List[Dict[str, Any]] = field(
        default_factory=list)
    agent_messages_sample: List[str] = field(default_factory=list)
    convergence_rounds: int = 0
    what_single_contract_would_miss: List[str] = field(
        default_factory=list)
    ground_truth: List[Dict] = field(default_factory=list)

    @property
    def n_interactions(self) -> int:
        """Total number of interactions from type breakdown."""
        if self.interaction_types:
            return sum(self.interaction_types.values())
        # Fallback: parse from graph_summary if available
        # Format: "dapp_name: 14C, 23E"
        import re
        m = re.search(r'(\d+)E', self.graph_summary)
        return int(m.group(1)) if m else 0

    def to_latex(self) -> str:
        """Generate a LaTeX paragraph for the paper (SS5.6)."""
        n_local = sum(self.local_findings_per_contract.values())
        n_cross = len(self.cross_contract_findings)

        lines = [
            f"\\paragraph{{Case Study: {_latex_escape(self.dapp_id)}}}",
            f"This DApp consists of {self.num_contracts} contracts "
            f"with {self.n_interactions} interactions.  "
            f"Local-only analysis identified {n_local} "
            f"intra-contract findings.  "
            f"\\system{{}}'s cross-contract reasoning detected "
            f"{n_cross} additional findings that local analysis "
            f"missed.",
        ]

        if self.cross_contract_findings:
            for f in self.cross_contract_findings[:3]:
                vtype = _latex_escape(f.get("type", ""))
                desc = _latex_escape(f.get("description", ""))
                lines.append(
                    f"\\emph{{{vtype}}}: {desc}")

        if self.convergence_rounds > 0:
            lines.append(
                f"The protocol converged in "
                f"{self.convergence_rounds} rounds.")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Module-level constants and helpers
# ═══════════════════════════════════════════════════════════════════════

# Severity weights for Equation (1) in SS3.4
_SEVERITY_WEIGHTS: Dict[str, float] = {
    "critical": 1.0,
    "high": 0.8,
    "medium": 0.5,
    "low": 0.3,
}

# Mapping from local vulnerability types to severity levels
_LOCAL_SEVERITY: Dict[str, str] = {
    "reentrancy": "high",
    "unchecked_return": "medium",
    "timestamp_dependency": "low",
    "missing_access_control": "high",
    "tx_origin": "medium",
    "integer_overflow": "high",
}


def _infer_local_severity(vuln_type: str) -> str:
    """Map a local vulnerability type to a severity level.

    Falls back to "medium" for unknown types.
    """
    return _LOCAL_SEVERITY.get(vuln_type, "medium")


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in a string."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text
