"""
Multi-Agent System for CrossGuard.

Each ContractAgent owns ONE contract, performs local analysis (Phase 1),
and participates in cross-contract reasoning (Phase 2).  All agents
share a single LLM backbone (DeepSeek-Coder-6.7B with QLoRA).

Pipeline per agent
------------------
Phase 1   LLM-driven local analysis + regex pattern fallback
          -> AgentSummary (findings, interface exposures, risk)
Phase 2   LLM-driven cross-contract reasoning + pattern fallback
          -> cross-contract vulnerability findings
Merge     LLM findings take priority; patterns fill type gaps

Design rationale (SS3.2 of paper): assigning one agent per contract
ensures each agent develops specialised knowledge of its contract's
state variables, access-control patterns, and external interface.
This knowledge feeds directly into cross-contract reasoning.

Author : [Anonymous for double-blind review]
Target : ACM Transactions on Software Engineering and Methodology
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .decomposer import ContractInfo, InteractionGraph, InteractionType

logger = logging.getLogger(__name__)


# ──────────────────── data structures ───────────────────────────────────────

@dataclass
class LocalFinding:
    """A single vulnerability finding from local (Phase 1) analysis."""
    function_name: str
    vulnerability_type: str
    confidence: float
    description: str = ""
    swc_id: int = -1
    source: str = ""          # "llm" or "pattern" — consumed by metrics


@dataclass
class InterfaceExposure:
    """A public/external function that forms part of the attack surface."""
    function_name: str
    visibility: str
    is_payable: bool = False
    modifies_state: bool = False
    has_access_control: bool = False
    parameters: str = ""


@dataclass
class AgentSummary:
    """Phase 1 output: per-contract analysis summary.

    Five elements (SS3.2):
      1. local_findings          detected vulnerabilities
      2. interface_exposures     public/external function surface
      3. risk_score              weighted combination of 1 + 2
      4. state_variables_shared  candidates for cross-contract sharing
      5. external_calls_made     targets of external calls
    """
    contract_name: str
    local_findings: List[LocalFinding] = field(default_factory=list)
    interface_exposures: List[InterfaceExposure] = field(default_factory=list)
    risk_score: float = 0.0
    state_variables_shared: List[str] = field(default_factory=list)
    external_calls_made: List[str] = field(default_factory=list)

    def to_message(self) -> str:
        """Serialise summary for inter-agent messaging (Phase 2)."""
        parts = [f"[Contract: {self.contract_name}] "
                 f"[Risk: {self.risk_score:.2f}]"]
        if self.local_findings:
            parts.append(f"Findings ({len(self.local_findings)}):")
            for f in self.local_findings[:5]:
                parts.append(
                    f"  - {f.function_name}: {f.vulnerability_type} "
                    f"(conf={f.confidence:.2f})")
        if self.interface_exposures:
            parts.append(
                f"Exposed interfaces ({len(self.interface_exposures)}):")
            for ie in self.interface_exposures[:5]:
                flags = []
                if ie.is_payable:
                    flags.append("payable")
                if ie.modifies_state:
                    flags.append("state-mod")
                if not ie.has_access_control:
                    flags.append("NO-ACCESS-CTRL")
                parts.append(
                    f"  - {ie.function_name}({ie.parameters}) "
                    f"[{', '.join(flags)}]")
        if self.external_calls_made:
            parts.append(
                f"External calls: "
                f"{', '.join(self.external_calls_made[:5])}")
        return "\n".join(parts)


# ──────────────────── prompt templates ──────────────────────────────────────

class PromptBuilder:
    """Prompt templates for LLM-based vulnerability analysis (SS3.2, SS3.3).

    Prompt text matches the templates reproduced verbatim in the paper.
    """

    LOCAL_ANALYSIS = (
        "You are a smart contract security auditor. Analyse the following "
        "Solidity contract for vulnerabilities.\n\n"
        "Contract: {contract_name}\n"
        "```solidity\n{source_code}\n```\n\n"
        "Respond ONLY with a JSON object:\n"
        '{{"findings": [{{"function": "...", "type": "...", '
        '"confidence": 0.0-1.0, "description": "..."}}], '
        '"risk_score": 0.0-1.0}}')

    CROSS_CONTRACT = (
        "You are a security expert analysing the INTERACTION between two "
        "smart contracts in the same DApp. Your task is to find "
        "vulnerabilities that arise from their interaction — NOT "
        "individual contract bugs.\n\n"
        "=== Contract A: {my_contract} ===\n{my_summary}\n\n"
        "=== Contract B: {other_contract} ===\n{other_summary}\n\n"
        "Interaction type: {interaction_type}\n"
        "Interaction details: {interaction_details}\n\n"
        "Consider these cross-contract vulnerability patterns:\n"
        "1. Cross-reentrancy: A calls B, B calls back to A before "
        "state update\n"
        "2. Approval abuse: Token approval in A exploited via B's "
        "transferFrom\n"
        "3. Oracle manipulation: Price feed from A manipulated to "
        "exploit B\n"
        "4. Proxy storage collision: Proxy A delegatecalls B with "
        "incompatible storage\n"
        "5. Flash loan attack path: Borrowed funds from A used to "
        "exploit B\n"
        "6. Shared state corruption: Both contracts modify shared "
        "state unsafely\n"
        "7. Callback chain: Chain of callbacks across contracts causes "
        "unexpected state\n"
        "8. Privilege escalation: Inherited permissions allow "
        "unintended access\n\n"
        "Respond ONLY with a JSON object:\n"
        '{{"cross_vulnerabilities": [{{"type": "pattern_name", '
        '"confidence": 0.0-1.0, "description": "...", '
        '"attack_path": ["ContractA", "ContractB"]}}], '
        '"risk_delta": 0.0-1.0}}')

    @classmethod
    def build_local(cls, ci: ContractInfo,
                    max_source_chars: int = 3500) -> str:
        """Build local analysis prompt.

        Source is truncated to *max_source_chars* characters, which
        approximates the 3 500-token budget described in SS3.2.
        """
        return cls.LOCAL_ANALYSIS.format(
            contract_name=ci.name,
            source_code=ci.source_code[:max_source_chars])

    @classmethod
    def build_cross(cls, my_name: str, my_summary: str,
                    other_name: str, other_summary: str,
                    itype: str, details: str) -> str:
        """Build cross-contract reasoning prompt for one graph edge."""
        return cls.CROSS_CONTRACT.format(
            my_contract=my_name, my_summary=my_summary,
            other_contract=other_name, other_summary=other_summary,
            interaction_type=itype, interaction_details=details)


# ──────────────────── LLM module ────────────────────────────────────────────

class LLMModule(nn.Module):
    """LLM backbone with QLoRA, lazy-loaded.

    A single LLMModule is shared across all agents in an AgentPool so
    that LoRA fine-tuning updates propagate to every agent.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._model = None
        self._tokenizer = None
        self._loaded = False
        if config.agent.use_fallback_mlp:
            self.fallback = nn.Sequential(
                nn.Linear(config.agent.fallback_hidden_dim,
                          config.agent.fallback_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.agent.fallback_hidden_dim, 2))
        else:
            self.fallback = None

    # ── public interface ───────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True when the LLM is loaded and ready for generation."""
        return self._loaded and self._model is not None

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from the LLM.  Returns '{}' if unavailable."""
        self._load_model()
        if self._model is None or self._tokenizer is None:
            return "{}"
        inputs = self._tokenizer(
            prompt, return_tensors="pt",
            truncation=True,
            max_length=self.config.agent.llm_max_length)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(0.01,
                                self.config.agent.agent_temperature),
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id)
        # Decode only the newly generated tokens
        return self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True)

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Return LoRA-adapted parameters for fine-tuning."""
        if self._model is None:
            self._load_model()
        if self._model:
            return [p for p in self._model.parameters() if p.requires_grad]
        return []

    # ── model loading ──────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load the quantised LLM and attach LoRA adapters."""
        if self._loaded or self.config.agent.use_fallback_mlp:
            return
        try:
            from transformers import (AutoModelForCausalLM,
                                      AutoTokenizer,
                                      BitsAndBytesConfig)
            from peft import get_peft_model, LoraConfig, TaskType

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.agent.llm_model, trust_remote_code=True)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.agent.llm_model,
                quantization_config=bnb,
                device_map="auto",
                trust_remote_code=True)
            if self.config.agent.use_lora:
                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.config.agent.lora_rank,
                    lora_alpha=self.config.agent.lora_alpha,
                    lora_dropout=self.config.agent.lora_dropout,
                    target_modules=["q_proj", "v_proj",
                                    "k_proj", "o_proj"])
                self._model = get_peft_model(self._model, lora_cfg)
            self._loaded = True
            n_train = sum(p.numel() for p in self._model.parameters()
                          if p.requires_grad)
            logger.info(f"LLM loaded: {self.config.agent.llm_model} "
                        f"({n_train:,} trainable params)")
        except ImportError as exc:
            logger.warning(
                f"transformers/peft unavailable ({exc}) — "
                f"falling back to MLP")
            self.config.agent.use_fallback_mlp = True
            self.fallback = nn.Sequential(
                nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 2))

    # ── JSON parsing ───────────────────────────────────────────────────

    @staticmethod
    def parse_json_response(text: str) -> Dict[str, Any]:
        """Robustly parse JSON from LLM output.

        Handles three common artefacts (SS3.2):
          1. Bare JSON
          2. JSON wrapped in markdown fences
          3. JSON embedded in explanatory prose
        """
        text = text.strip()
        # Attempt 1: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Attempt 2: strip markdown fences
        cleaned = re.sub(r'```(?:json)?\s*', '', text)
        cleaned = re.sub(r'\s*```', '', cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Attempt 3: extract first balanced { ... } block
        depth, start = 0, -1
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        start = -1  # keep scanning for next block
        return {}


# ──────────────────── contract agent ────────────────────────────────────────

class ContractAgent:
    """Agent specialised for one contract.

    Uses the shared LLM for local analysis (Phase 1) and cross-contract
    reasoning (Phase 2).  Pattern-based detection is the fallback when
    the LLM is unavailable or its output cannot be parsed.
    """

    def __init__(self, agent_id: str, contract: ContractInfo,
                 llm: Optional[LLMModule] = None, config=None):
        self.agent_id = agent_id
        self.contract = contract
        self.llm = llm
        self.config = config
        self.summary: Optional[AgentSummary] = None
        self.received_messages: List[Dict[str, Any]] = []
        self.cross_contract_findings: List[Dict[str, Any]] = []

    # ================================================================
    # Phase 1: local analysis
    # ================================================================

    def run_local_analysis(self) -> AgentSummary:
        """Analyse the owned contract using LLM + pattern matching.

        Merge strategy (SS3.2): for each vulnerability type, if the LLM
        produced a finding the pattern-based finding for that type is
        suppressed.  Pattern findings are kept only for types the LLM
        did not report, providing coverage when LLM output is empty or
        unparseable.
        """
        pattern_findings = self._pattern_local_vulns()
        pattern_exposures = self._analyse_interfaces()

        llm_findings = self._llm_local_analysis()

        # Merge: LLM findings first (priority), patterns fill type gaps
        seen_types: Set[str] = set()
        merged: List[LocalFinding] = []
        for f in llm_findings:
            if f.vulnerability_type not in seen_types:
                merged.append(f)
                seen_types.add(f.vulnerability_type)
        for f in pattern_findings:
            if f.vulnerability_type not in seen_types:
                merged.append(f)
                seen_types.add(f.vulnerability_type)

        self.summary = AgentSummary(
            contract_name=self.contract.name,
            local_findings=merged,
            interface_exposures=pattern_exposures,
            risk_score=self._compute_risk(merged, pattern_exposures),
            state_variables_shared=self._shared_state_candidates(),
            external_calls_made=self._ext_call_targets())
        return self.summary

    def _llm_local_analysis(self) -> List[LocalFinding]:
        """LLM-driven local analysis with retries on parse failure.

        Paper (SS3.2): "Parsing failures trigger up to two retries
        before the agent falls back to an empty finding set."
        """
        if not self.llm:
            return []
        if self.config and self.config.agent.use_fallback_mlp:
            return []
        if self.config and self.config.agent.local_analysis_max_tokens <= 0:
            return []                   # disabled (cross-only ablation)

        max_tokens = (self.config.agent.local_analysis_max_tokens
                      if self.config else 1024)
        retries = (self.config.agent.cross_reasoning_retries
                   if self.config else 2)
        prompt = PromptBuilder.build_local(self.contract)

        for attempt in range(retries):
            try:
                raw = self.llm.generate(prompt, max_tokens)
                parsed = LLMModule.parse_json_response(raw)
                findings: List[LocalFinding] = []
                for f in parsed.get("findings", []):
                    findings.append(LocalFinding(
                        function_name=f.get("function", ""),
                        vulnerability_type=f.get("type", "unknown"),
                        confidence=_clamp_conf(f.get("confidence", 0.5)),
                        description=f.get("description", ""),
                        source="llm"))
                if findings:
                    return findings
            except Exception as exc:
                logger.debug(
                    f"LLM local analysis attempt {attempt + 1} "
                    f"failed for {self.contract.name}: {exc}")
        return []

    # ================================================================
    # Phase 2: cross-contract reasoning
    # ================================================================

    def process_cross_contract(
            self,
            neighbour_summary: AgentSummary,
            interaction: Any) -> List[Dict[str, Any]]:
        """Cross-contract reasoning: LLM primary, pattern fallback.

        For each edge in the interaction graph the agent reasons about
        vulnerabilities arising from the interaction with the neighbour.

        Returns dicts with keys:
          type, path, confidence, description, source ("llm"/"pattern")
        """
        cross_vulns: List[Dict[str, Any]] = []

        # PRIMARY: LLM-driven reasoning
        if (self.llm and self.config
                and self.config.agent.use_llm_cross_reasoning
                and not self.config.agent.use_fallback_mlp):
            cross_vulns.extend(
                self._llm_cross_reasoning(neighbour_summary, interaction))

        # FALLBACK: pattern-based detection (supplements LLM)
        pattern_vulns = self._pattern_cross_reasoning(
            neighbour_summary, interaction)

        # Merge: LLM types take priority; patterns fill gaps
        llm_types = {v.get("type", "") for v in cross_vulns}
        for pv in pattern_vulns:
            if pv.get("type", "") not in llm_types:
                cross_vulns.append(pv)

        self.cross_contract_findings.extend(cross_vulns)
        return cross_vulns

    def _llm_cross_reasoning(
            self,
            ns: AgentSummary,
            interaction: Any) -> List[Dict[str, Any]]:
        """Use LLM to reason about cross-contract vulnerabilities.

        Paper (SS3.3): "If the LLM's response fails to parse after up
        to two retries, the agent falls back to the pattern-based
        cross-contract detector."
        """
        if self.summary is None:
            logger.warning(
                f"Agent {self.agent_id}: cross reasoning invoked "
                f"before local analysis — skipping LLM call")
            return []

        itype = (interaction.interaction_type.name
                 if hasattr(interaction, 'interaction_type')
                 else "EXTERNAL_CALL")
        details = str(getattr(interaction, 'details', {}))

        prompt = PromptBuilder.build_cross(
            self.contract.name, self.summary.to_message(),
            ns.contract_name, ns.to_message(),
            itype, details)

        max_tokens = (self.config.agent.cross_contract_max_tokens
                      if self.config else 768)
        retries = (self.config.agent.cross_reasoning_retries
                   if self.config else 2)

        for attempt in range(retries):
            try:
                raw = self.llm.generate(prompt, max_tokens)
                parsed = LLMModule.parse_json_response(raw)
                vulns: List[Dict[str, Any]] = []
                for cv in parsed.get("cross_vulnerabilities", []):
                    vulns.append({
                        "type": cv.get("type", "unknown_cross"),
                        "path": cv.get("attack_path",
                                       [self.contract.name,
                                        ns.contract_name]),
                        "confidence": _clamp_conf(
                            cv.get("confidence", 0.5)),
                        "description": cv.get("description", ""),
                        "source": "llm"})
                if vulns:
                    return vulns
            except Exception as exc:
                logger.debug(
                    f"LLM cross reasoning attempt {attempt + 1} "
                    f"failed: {exc}")
        return []

    # ================================================================
    # Pattern-based cross-contract detection (fallback)
    # ================================================================

    def _pattern_cross_reasoning(
            self,
            ns: AgentSummary,
            interaction: Any) -> List[Dict[str, Any]]:
        """Structural pattern checkers for six cross-contract families.

        Paper (SS3.3): "For pattern-based fallback, six of the eight
        families have explicit structural checkers; flash loan attack
        path and privilege escalation are detected only through LLM
        reasoning."

        Confidence scores are conservative (0.5-0.8) to reflect
        patterns' inability to reason about semantic context.

        Each checker is gated on relevant edge types to avoid firing
        on structurally inappropriate edges.
        """
        vulns: List[Dict[str, Any]] = []
        itype = getattr(interaction, 'interaction_type', None)

        # (checker, vuln_type, confidence, swc_id, valid_edge_types)
        checks = [
            (self._check_cross_reentrancy,
             "cross_reentrancy", 0.7, 107,
             {InteractionType.EXTERNAL_CALL}),

            (self._check_approval_abuse,
             "approval_abuse", 0.65, 104,
             {InteractionType.EXTERNAL_CALL,
              InteractionType.INTERFACE_DEP}),

            (self._check_oracle_manipulation,
             "oracle_manipulation", 0.6, 114,
             {InteractionType.EXTERNAL_CALL,
              InteractionType.INTERFACE_DEP}),

            (self._check_proxy_collision,
             "proxy_storage_collision", 0.8, 112,
             {InteractionType.PROXY_DELEGATE}),

            (self._check_callback_chain,
             "callback_chain", 0.55, 107,
             {InteractionType.EXTERNAL_CALL}),

            (self._check_shared_state,
             "shared_state_corruption", 0.6, -1,
             {InteractionType.STATE_DEPENDENCY,
              InteractionType.EXTERNAL_CALL}),
        ]

        for checker, vtype, conf, swc, valid_edges in checks:
            if itype is not None and itype not in valid_edges:
                continue
            if checker(ns, interaction):
                vulns.append({
                    "type": vtype,
                    "path": [self.contract.name, ns.contract_name],
                    "confidence": conf,
                    "swc_id": swc,
                    "source": "pattern"})
        return vulns

    # ── individual checkers ────────────────────────────────────────────

    def _check_cross_reentrancy(self, ns: AgentSummary,
                                interaction: Any) -> bool:
        """Cross-reentrancy: A calls B via low-level call (.call/.send),
        B can call back into A's state-modifying function before A
        updates state.

        Checks:
          1. A makes a low-level external call that can transfer
             control to B (method is .call or .send)
          2. B exposes at least one state-modifying public function
             without a reentrancy guard or access control
        """
        has_callback_risk = any(
            c.get("method") in ("call", "send")
            for c in self.contract.external_calls)
        if not has_callback_risk:
            return False
        return any(ie.modifies_state and not ie.has_access_control
                   for ie in ns.interface_exposures)

    def _check_approval_abuse(self, ns: AgentSummary,
                              interaction: Any) -> bool:
        """Approval abuse: token approval in A exploited via B's
        transferFrom.
        """
        has_approve = bool(re.search(
            r'\bapprove\s*\(', self.contract.source_code))
        has_transfer_from = any(
            ie.function_name == "transferFrom"
            for ie in ns.interface_exposures)
        return has_approve and has_transfer_from

    def _check_oracle_manipulation(self, ns: AgentSummary,
                                   interaction: Any) -> bool:
        """Oracle manipulation: price feed consumed without staleness
        or deviation check.
        """
        price_read = bool(re.search(
            r'getPrice|latestAnswer|getRoundData|latestRoundData',
            self.contract.source_code))
        price_influence = any(
            re.search(r'price|rate|oracle|feed',
                      ie.function_name, re.IGNORECASE)
            for ie in ns.interface_exposures)
        return price_read and price_influence

    def _check_proxy_collision(self, ns: AgentSummary,
                               interaction: Any) -> bool:
        """Proxy storage collision: proxy delegatecalls into an
        implementation whose storage layout may be incompatible.
        """
        itype = getattr(interaction, 'interaction_type', None)
        return itype == InteractionType.PROXY_DELEGATE

    def _check_callback_chain(self, ns: AgentSummary,
                              interaction: Any) -> bool:
        """Callback chain: A calls B, and B itself makes further
        external calls, creating a potential multi-hop chain.

        Requires B to have >= 2 external call targets (suggesting it
        is a routing hub that forwards execution), not merely that
        both contracts have any external calls.
        """
        if self.summary is None:
            return False
        my_ext = len(self.summary.external_calls_made)
        other_ext = len(ns.external_calls_made)
        return my_ext >= 1 and other_ext >= 2

    def _check_shared_state(self, ns: AgentSummary,
                            interaction: Any) -> bool:
        """Shared state corruption: two contracts modify the same
        storage variable without mutual exclusion.
        """
        if self.summary is None:
            return False
        overlap = (set(self.summary.state_variables_shared)
                   & set(ns.state_variables_shared))
        return len(overlap) > 0

    # ================================================================
    # Local vulnerability patterns (Phase 1 fallback)
    # ================================================================

    def _pattern_local_vulns(self) -> List[LocalFinding]:
        """Regex-based vulnerability detection for six families.

        Each rule has a positive pattern (potential vulnerability) and
        an optional negative pattern (mitigation).  Confidence scores
        are conservative (0.5-0.8).

        Paper (SS3.2): "Six vulnerability families are covered."
        """
        findings: List[LocalFinding] = []
        src = self.contract.source_code

        # (positive_re, negative_re, type, confidence, swc_id)
        patterns = [
            (r'\.call\{value|\.call\.value|\.send\(|\.transfer\(',
             r'nonReentrant|ReentrancyGuard',
             "reentrancy", 0.7, 107),

            (r'\.call\s*\(|\.call\{',
             r'\(bool\s+\w+[^)]*\)\s*=\s*\S+\.call'
             r'|require\s*\([^)]*\.call',
             "unchecked_return", 0.6, 104),

            (r'block\.timestamp\s*[<>=!]|now\s*[<>=!]',
             None,
             "timestamp_dependency", 0.5, 116),

            (r'selfdestruct\s*\(|\.transfer\s*\(|\.call\{value',
             r'onlyOwner|require\s*\(\s*msg\.sender\s*=='
             r'|onlyAdmin',
             "missing_access_control", 0.6, 105),

            (r'tx\.origin',
             None,
             "tx_origin", 0.8, 115),
        ]

        for pos_re, neg_re, vtype, conf, swc in patterns:
            if re.search(pos_re, src):
                if neg_re is None or not re.search(neg_re, src):
                    findings.append(LocalFinding(
                        function_name="",
                        vulnerability_type=vtype,
                        confidence=conf,
                        swc_id=swc,
                        source="pattern"))

        # Integer overflow: only for Solidity < 0.8.0 without SafeMath
        version = self._detect_compiler_version()
        if version is not None and version < (0, 8, 0):
            if not re.search(r'SafeMath|using\s+SafeMath', src):
                if re.search(r'[\+\-\*]', src):
                    findings.append(LocalFinding(
                        function_name="",
                        vulnerability_type="integer_overflow",
                        confidence=0.6,
                        swc_id=101,
                        source="pattern"))

        return findings

    # ================================================================
    # Interface analysis
    # ================================================================

    def _analyse_interfaces(self) -> List[InterfaceExposure]:
        """Characterise the contract's public/external interface.

        For each public/external function, determine:
          - is_payable:         declared as payable
          - modifies_state:     not declared view/pure (compiler-enforced)
          - has_access_control: modifier or require(msg.sender) present

        Uses per-function source extraction to avoid whole-contract
        false positives (e.g., marking all functions as access-controlled
        because one function has onlyOwner).
        """
        exposures: List[InterfaceExposure] = []
        for fn in self.contract.functions:
            if fn["visibility"] not in ("public", "external"):
                continue

            fn_src = self._get_function_source(fn["name"])

            # ── access control ──
            has_ac = False
            if fn_src:
                # Modifiers in the function declaration (between ) and {)
                decl_m = re.match(
                    r'function\s+\w+\s*\([^)]*\)(.*?)\{',
                    fn_src, re.DOTALL)
                if decl_m:
                    modifiers_text = decl_m.group(1)
                    has_ac = bool(re.search(
                        r'onlyOwner|onlyAdmin|onlyRole|whenNotPaused'
                        r'|onlyAuthorized|onlyMinter|onlyGovernance',
                        modifiers_text))
                # require(msg.sender ==) checks inside the body
                if not has_ac:
                    has_ac = bool(re.search(
                        r'require\s*\(\s*msg\.sender\s*=='
                        r'|_checkOwner\s*\('
                        r'|_checkRole\s*\(',
                        fn_src))

            # ── state modification: view/pure cannot modify ──
            is_view_pure = False
            if fn_src:
                decl_m = re.match(
                    r'function\s+\w+\s*\([^)]*\)(.*?)\{',
                    fn_src, re.DOTALL)
                if decl_m:
                    is_view_pure = bool(re.search(
                        r'\b(view|pure)\b', decl_m.group(1)))

            exposures.append(InterfaceExposure(
                function_name=fn["name"],
                visibility=fn["visibility"],
                is_payable=fn.get("is_payable", False),
                modifies_state=not is_view_pure,
                has_access_control=has_ac,
                parameters=fn.get("params", "")))
        return exposures

    # ================================================================
    # Helpers
    # ================================================================

    def _get_function_source(self, fn_name: str) -> str:
        """Extract the source of a specific function from contract code.

        Returns signature + body (including braces), or empty string
        if the function cannot be located.
        """
        pattern = rf'function\s+{re.escape(fn_name)}\s*\([^)]*\)[^{{]*\{{'
        m = re.search(pattern, self.contract.source_code)
        if not m:
            return ""
        start = m.start()
        brace_pos = m.end() - 1          # position of opening {
        depth = 0
        limit = min(brace_pos + 15_000, len(self.contract.source_code))
        for i in range(brace_pos, limit):
            ch = self.contract.source_code[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return self.contract.source_code[start:i + 1]
        return self.contract.source_code[start:start + 3000]

    def _detect_compiler_version(self) -> Optional[Tuple[int, ...]]:
        """Parse Solidity pragma into a numeric tuple for comparison.

        Returns (major, minor, patch) or None.  Using a tuple avoids
        the lexicographic pitfall where '0.10.0' < '0.8.0' is True
        under string comparison.
        """
        m = re.search(
            r'pragma\s+solidity\s+[\^~>=<]*\s*([\d.]+)',
            self.contract.source_code)
        if not m:
            return None
        parts: List[int] = []
        for segment in m.group(1).split("."):
            try:
                parts.append(int(segment))
            except ValueError:
                parts.append(0)
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts)

    def _compute_risk(self, findings: List[LocalFinding],
                      exposures: List[InterfaceExposure]) -> float:
        """Compute contract risk score in [0, 1].

        Weighted combination of finding severity and interface exposure
        risk, normalised so the score does not saturate with count.
        """
        # Finding risk: average confidence, scaled by count
        if findings:
            avg_conf = (sum(f.confidence for f in findings)
                        / len(findings))
            # Saturates at 3 findings: one finding contributes less
            # than three, but four is not much more than three.
            count_factor = min(1.0, len(findings) / 3.0)
            finding_risk = avg_conf * count_factor
        else:
            finding_risk = 0.0

        # Exposure risk: fraction of dangerous public functions
        if exposures:
            dangerous = sum(
                1 for e in exposures
                if not e.has_access_control
                and (e.is_payable or e.modifies_state))
            exposure_risk = dangerous / len(exposures)
        else:
            exposure_risk = 0.0

        return min(1.0, 0.6 * finding_risk + 0.4 * exposure_risk)

    def _shared_state_candidates(self) -> List[str]:
        """State variables likely to be shared across contracts.

        Types that commonly appear in cross-contract storage:
        mappings, addresses, and uint256 balances.
        """
        return [sv["name"] for sv in self.contract.state_variables
                if sv["type"] in ("mapping", "address", "uint256")]

    def _ext_call_targets(self) -> List[str]:
        """Sorted names of external call targets."""
        return sorted(set(
            c.get("target_var", c.get("interface", "?"))
            for c in self.contract.external_calls))


# ──────────────────── agent pool ────────────────────────────────────────────

class AgentPool:
    """Manages contract agents for a single DApp analysis.

    When the DApp has more contracts than *max_agents_per_dapp*,
    contracts are ranked by interaction degree (sum of in + out edges)
    and the top-k are retained.  This ensures the most interconnected
    contracts receive dedicated agents (SS3.2).
    """

    def __init__(self, graph: InteractionGraph, config, llm=None):
        self.graph = graph
        self.config = config
        self.llm = llm
        self.agents: Dict[str, ContractAgent] = {}
        self._create()

    def _create(self):
        contracts = sorted(self.graph.contracts.keys())
        mx = self.config.agent.max_agents_per_dapp

        if len(contracts) > mx:
            degree: Dict[str, int] = defaultdict(int)
            for e in self.graph.interactions:
                degree[e.source] += 1
                degree[e.target] += 1
            contracts = sorted(
                contracts,
                key=lambda c: degree.get(c, 0),
                reverse=True)[:mx]
            logger.info(
                f"AgentPool: pruned {self.graph.num_contracts} "
                f"contracts to {mx} by interaction degree")

        for cn in contracts:
            self.agents[cn] = ContractAgent(
                agent_id=f"agent_{cn}",
                contract=self.graph.contracts[cn],
                llm=self.llm,
                config=self.config)
        logger.info(
            f"AgentPool: {len(self.agents)} agents for "
            f"{self.graph.dapp_name}")

    def run_local_analysis(self) -> Dict[str, AgentSummary]:
        """Run Phase 1 local analysis for every agent."""
        return {cn: agent.run_local_analysis()
                for cn, agent in self.agents.items()}

    def get_agent(self, cn: str) -> Optional[ContractAgent]:
        return self.agents.get(cn)


# ──────────────────── module-level helpers ──────────────────────────────────

def _clamp_conf(raw) -> float:
    """Clamp a raw confidence value from LLM output to [0, 1]."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, v))
