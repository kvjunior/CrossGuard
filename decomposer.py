"""
DApp decomposition and interaction graph for CrossGuard.

Constructs G = (V, E) where V = contracts, E = typed interactions.
Six edge types (SS3.2):
  EXTERNAL_CALL, INHERITANCE, INTERFACE_DEP,
  STATE_DEPENDENCY, EVENT_DEPENDENCY, PROXY_DELEGATE

The parser is regex-based rather than AST-based (SS3.2): "While less
precise than a full compiler front end, this approach handles
incomplete source files and mixed compiler versions that are common
in DAppSCAN's real-world corpus without requiring a working Solidity
toolchain."

Author : [Anonymous for double-blind review]
Target : ACM Transactions on Software Engineering and Methodology
"""

from __future__ import annotations

import logging
import random as _random_module
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Core data structures
# ═══════════════════════════════════════════════════════════════════════

class InteractionType(Enum):
    """Six categories of inter-contract coupling (SS3.2)."""
    EXTERNAL_CALL = auto()
    INHERITANCE = auto()
    INTERFACE_DEP = auto()
    STATE_DEPENDENCY = auto()
    EVENT_DEPENDENCY = auto()
    PROXY_DELEGATE = auto()


@dataclass
class ContractInfo:
    """Parsed structural representation of a single Solidity contract.

    Fields are populated by DAppDecomposer._extract_contracts().
    The corrected agents.py accesses: name, source_code, functions,
    state_variables, external_calls, is_proxy, file_path, loc.
    """
    name: str
    file_path: str
    source_code: str
    functions: List[Dict[str, Any]] = field(default_factory=list)
    state_variables: List[Dict[str, Any]] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    inherits_from: List[str] = field(default_factory=list)
    interfaces_used: List[str] = field(default_factory=list)
    external_calls: List[Dict[str, str]] = field(default_factory=list)
    compiler_version: str = ""
    is_proxy: bool = False
    is_library: bool = False
    is_factory: bool = False
    loc: int = 0

    @property
    def interface_points(self) -> List[str]:
        """Names of public/external functions (the contract's surface)."""
        return [f["name"] for f in self.functions
                if f.get("visibility") in ("public", "external")]

    def summary_text(self, max_len: int = 800) -> str:
        """Compact text summary for LLM prompts."""
        parts = [f"Contract {self.name} ({self.loc} LOC)"]
        if self.inherits_from:
            parts.append(f"Inherits: {', '.join(self.inherits_from)}")
        pub_fns = self.interface_points
        if pub_fns:
            parts.append(
                f"Public functions: {', '.join(pub_fns[:10])}")
        payable = [f["name"] for f in self.functions
                   if f.get("is_payable")]
        if payable:
            parts.append(f"Payable: {', '.join(payable)}")
        if self.state_variables:
            names = [v["name"] for v in self.state_variables[:8]]
            parts.append(f"State vars: {', '.join(names)}")
        if self.is_proxy:
            parts.append("FLAGS: PROXY_CONTRACT")
        if self.is_factory:
            parts.append("FLAGS: FACTORY_CONTRACT")
        return "\n".join(parts)[:max_len]


@dataclass
class Interaction:
    """A single typed, directed edge in the interaction graph."""
    source: str
    target: str
    interaction_type: InteractionType
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def __str__(self) -> str:
        return (f"{self.source} --[{self.interaction_type.name}]--> "
                f"{self.target}")

    @property
    def _dedup_key(self) -> Tuple:
        """Key for deduplication (ignores details and confidence)."""
        return (self.source, self.target, self.interaction_type)


@dataclass
class InteractionGraph:
    """G = (V, E): contracts as nodes, typed interactions as edges.

    The graph is built by DAppDecomposer.decompose() and consumed by
    AgentPool (to assign agents), CrossContractReasoningProtocol
    (to route messages), and the visualiser (for paper figures).
    """
    contracts: Dict[str, ContractInfo] = field(default_factory=dict)
    interactions: List[Interaction] = field(default_factory=list)
    dapp_name: str = ""

    @property
    def num_contracts(self) -> int:
        return len(self.contracts)

    @property
    def num_interactions(self) -> int:
        return len(self.interactions)

    def get_neighbours(self, cname: str,
                       ) -> List[Tuple[str, InteractionType]]:
        """Return all neighbours of a contract in the graph."""
        nb: List[Tuple[str, InteractionType]] = []
        for e in self.interactions:
            if e.source == cname:
                nb.append((e.target, e.interaction_type))
            elif e.target == cname:
                nb.append((e.source, e.interaction_type))
        return nb

    def get_interactions_between(self, c1: str, c2: str,
                                ) -> List[Interaction]:
        """Return all edges between two contracts (either direction)."""
        return [e for e in self.interactions
                if {e.source, e.target} == {c1, c2}]

    def get_call_chains(self, max_length: int = 5,
                        max_chains: int = 500) -> List[List[str]]:
        """Enumerate simple paths along EXTERNAL_CALL / PROXY_DELEGATE
        edges, up to *max_length* hops.

        Used by the multi-hop attack-path detector (SS3.3).
        Capped at *max_chains* to prevent combinatorial explosion on
        highly connected DApps.
        """
        adj: Dict[str, Set[str]] = defaultdict(set)
        for e in self.interactions:
            if e.interaction_type in (InteractionType.EXTERNAL_CALL,
                                      InteractionType.PROXY_DELEGATE):
                adj[e.source].add(e.target)
        chains: List[List[str]] = []
        for start in adj:
            self._dfs(start, [start], adj, chains,
                      max_length, max_chains)
            if len(chains) >= max_chains:
                break
        return chains

    def _dfs(self, node: str, path: List[str],
             adj: Dict[str, Set[str]],
             chains: List[List[str]],
             max_len: int, max_chains: int):
        """Depth-first path enumeration with chain-count cap."""
        if len(chains) >= max_chains:
            return
        if len(path) > max_len:
            return
        if len(path) > 1:
            chains.append(list(path))
        for nxt in adj.get(node, ()):
            if nxt not in path:
                path.append(nxt)
                self._dfs(nxt, path, adj, chains,
                          max_len, max_chains)
                path.pop()

    def to_adjacency(self) -> Dict[str, List[str]]:
        """Undirected adjacency list representation."""
        adj: Dict[str, Set[str]] = defaultdict(set)
        for e in self.interactions:
            adj[e.source].add(e.target)
            adj[e.target].add(e.source)
        return {k: sorted(v) for k, v in adj.items()}

    def summary(self) -> str:
        """Human-readable summary for logging."""
        tc: Dict[str, int] = defaultdict(int)
        for e in self.interactions:
            tc[e.interaction_type.name] += 1
        parts = [f"{self.dapp_name}: "
                 f"{self.num_contracts}C, "
                 f"{self.num_interactions}E"]
        for t, c in sorted(tc.items()):
            parts.append(f"  {t}: {c}")
        return "\n".join(parts)

    def deduplicate_edges(self):
        """Remove duplicate edges (same source, target, type)."""
        seen: Set[Tuple] = set()
        unique: List[Interaction] = []
        for e in self.interactions:
            key = e._dedup_key
            if key not in seen:
                seen.add(key)
                unique.append(e)
        removed = len(self.interactions) - len(unique)
        if removed > 0:
            logger.debug(
                f"Deduplicated {removed} edges in "
                f"{self.dapp_name}")
        self.interactions = unique

    # ── visualisation ──────────────────────────────────────────────

    def visualise(self, output_path: str,
                  layout: str = "spring"):
        """Generate interaction graph figure for paper (SS3.2).

        Uses a deterministic local RNG to avoid perturbing the global
        random state.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.warning(
                "matplotlib unavailable for graph visualisation")
            return

        nodes = sorted(self.contracts.keys())
        n = len(nodes)
        if n < 2:
            logger.debug(
                f"Skipping graph figure for {self.dapp_name}: "
                f"only {n} contract(s)")
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # ── compute layout ─────────────────────────────────────────
        if layout == "circular":
            import math
            pos = {
                name: (math.cos(2 * math.pi * i / n),
                       math.sin(2 * math.pi * i / n))
                for i, name in enumerate(nodes)}
        else:
            # Simple force-directed layout with local RNG
            rng = _random_module.Random(42)
            pos = {name: (rng.gauss(0, 1), rng.gauss(0, 1))
                   for name in nodes}
            for _ in range(50):
                for a in nodes:
                    fx, fy = 0.0, 0.0
                    # Repulsive force between all pairs
                    for b in nodes:
                        if a == b:
                            continue
                        dx = pos[a][0] - pos[b][0]
                        dy = pos[a][1] - pos[b][1]
                        d = max(0.01, (dx**2 + dy**2)**0.5)
                        fx += 0.1 * dx / d
                        fy += 0.1 * dy / d
                    # Attractive force along edges
                    for e in self.interactions:
                        if e.source == a:
                            other = e.target
                        elif e.target == a:
                            other = e.source
                        else:
                            continue
                        dx = pos[other][0] - pos[a][0]
                        dy = pos[other][1] - pos[a][1]
                        fx += 0.05 * dx
                        fy += 0.05 * dy
                    pos[a] = (pos[a][0] + fx * 0.1,
                              pos[a][1] + fy * 0.1)

        # ── colour scheme ──────────────────────────────────────────
        edge_colors = {
            InteractionType.EXTERNAL_CALL: "#F44336",
            InteractionType.INHERITANCE: "#2196F3",
            InteractionType.INTERFACE_DEP: "#4CAF50",
            InteractionType.STATE_DEPENDENCY: "#FF9800",
            InteractionType.EVENT_DEPENDENCY: "#9C27B0",
            InteractionType.PROXY_DELEGATE: "#795548",
        }

        # ── draw edges ─────────────────────────────────────────────
        for e in self.interactions:
            x0, y0 = pos[e.source]
            x1, y1 = pos[e.target]
            c = edge_colors.get(e.interaction_type, "#888")
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=c,
                                lw=1.5, alpha=0.7))

        # ── draw nodes ─────────────────────────────────────────────
        for name in nodes:
            x, y = pos[name]
            ci = self.contracts[name]
            nc = "#FFEBEE" if ci.is_proxy else "#E3F2FD"
            ax.scatter(x, y, s=600, c=nc, edgecolors="#333",
                       zorder=5, linewidths=1.5)
            label = (name[:15] + "...") if len(name) > 15 else name
            ax.annotate(label, (x, y), ha="center", va="center",
                        fontsize=7, zorder=6)

        # ── legend ─────────────────────────────────────────────────
        patches = [
            mpatches.Patch(
                color=c,
                label=t.name.replace("_", " ").title())
            for t, c in edge_colors.items()]
        ax.legend(handles=patches, loc="upper left", fontsize=7)
        ax.set_title(
            f"Interaction Graph: {self.dapp_name}", fontsize=11)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Graph figure saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# DApp decomposer
# ═══════════════════════════════════════════════════════════════════════

class DAppDecomposer:
    """Parse DApp source files and construct the interaction graph.

    Pipeline (SS3.2):
      1. Extract all contract/library/interface definitions
      2. Parse internal structure (functions, state vars, calls, etc.)
      3. Build typed interaction edges across six categories
      4. Deduplicate edges
    """

    def __init__(self, config):
        self.config = config

    def decompose(self, dapp_files: Dict[str, str],
                  dapp_name: str = "") -> InteractionGraph:
        """Decompose a DApp into contracts and build the interaction graph.

        Parameters
        ----------
        dapp_files : dict
            Mapping from filename to Solidity source code.
        dapp_name : str
            Identifier for logging and visualisation.

        Returns
        -------
        InteractionGraph
            G = (V, E) where V = contracts, E = typed interactions.
        """
        graph = InteractionGraph(dapp_name=dapp_name)

        # ── extract contracts from all source files ────────────────
        for fname, source in dapp_files.items():
            for ci in self._extract_contracts(source, fname):
                if len(graph.contracts) >= \
                        self.config.decomposer.max_contracts_per_dapp:
                    break
                graph.contracts[ci.name] = ci

        # ── detect interactions ────────────────────────────────────
        names = set(graph.contracts.keys())
        detectors = [
            (self.config.decomposer.detect_inheritance,
             self._detect_inheritance),
            (self.config.decomposer.detect_external_calls,
             self._detect_external_calls),
            (self.config.decomposer.detect_interface_deps,
             self._detect_interface_deps),
            (self.config.decomposer.detect_state_deps,
             self._detect_state_deps),
            (self.config.decomposer.detect_event_deps,
             self._detect_event_deps),
            (self.config.decomposer.detect_proxy_patterns,
             self._detect_proxy_patterns),
        ]
        for enabled, fn in detectors:
            if enabled:
                fn(graph, names)

        # ── deduplicate ────────────────────────────────────────────
        graph.deduplicate_edges()

        logger.info(f"Decomposed: {graph.summary()}")
        return graph

    def decompose_single(self, source: str,
                         cid: str = "") -> InteractionGraph:
        """Convenience: decompose a single source file."""
        return self.decompose(
            {cid or "main.sol": source}, dapp_name=cid)

    # ================================================================
    # Contract extraction (regex-based parser)
    # ================================================================

    def _extract_contracts(self, source: str,
                           fname: str) -> List[ContractInfo]:
        """Extract all contract/library/interface definitions from source.

        Known limitation: the brace-matching approach does not handle
        { or } inside string literals or comments.  This is acceptable
        for the DAppSCAN corpus where most source files are well-formed.
        """
        contracts: List[ContractInfo] = []
        for m in re.finditer(
                r'(contract|library|interface)\s+(\w+)'
                r'\s*(?:is\s+([^{]+))?\s*\{',
                source):
            kind = m.group(1)
            name = m.group(2)
            inh_str = m.group(3) or ""
            body = self._extract_body(source, m.end() - 1)

            ci = ContractInfo(
                name=name,
                file_path=fname,
                source_code=body,
                loc=body.count("\n") + 1,
                is_library=(kind == "library"))

            # Inheritance list
            if inh_str:
                ci.inherits_from = [
                    s.strip().split("(")[0].strip()
                    for s in inh_str.split(",")
                    if s.strip()]

            ci.functions = self._extract_functions(body)
            ci.state_variables = self._extract_state_vars(body)
            ci.modifiers = re.findall(r'modifier\s+(\w+)', body)
            ci.events = re.findall(r'event\s+(\w+)', body)
            ci.external_calls = self._extract_ext_calls(body)

            # Interface references: names starting with I followed
            # by an uppercase letter (Solidity convention)
            ci.interfaces_used = sorted(set(
                re.findall(r'\bI([A-Z]\w+)\b', body)))
            # Store with the I prefix for downstream matching
            ci.interfaces_used = [
                f"I{n}" for n in ci.interfaces_used]

            # Proxy detection
            ci.is_proxy = bool(re.search(
                r'delegatecall|_implementation\b|upgradeTo\b'
                r'|_fallback\b|fallback\s*\(\s*\)',
                body))

            # Factory detection
            ci.is_factory = bool(re.search(
                r'\bnew\s+\w+\s*\(|\.creationCode\b|create2\b',
                body))

            # Compiler version
            v = re.search(
                r'pragma\s+solidity\s+[\^~>=<]*\s*([\d.]+)',
                source)
            if v:
                ci.compiler_version = v.group(1)

            contracts.append(ci)
        return contracts

    def _extract_body(self, src: str, brace_start: int) -> str:
        """Extract balanced-brace body starting at the opening {.

        Known limitation: does not handle braces inside string literals
        or block comments.  This is documented in the paper (SS3.2).
        """
        depth = 0
        limit = min(brace_start + 50_000, len(src))
        for i in range(brace_start, limit):
            if src[i] == '{':
                depth += 1
            elif src[i] == '}':
                depth -= 1
                if depth == 0:
                    return src[brace_start:i + 1]
        return src[brace_start:brace_start + 5000]

    def _extract_functions(self, body: str) -> List[Dict[str, Any]]:
        """Extract function signatures from a contract body.

        Uses re.DOTALL so the pattern matches multi-line declarations,
        which are common when functions have many parameters or
        multiple modifiers.
        """
        fns: List[Dict[str, Any]] = []
        for m in re.finditer(
                r'function\s+(\w+)\s*\(([^)]*)\)\s*'
                r'((?:public|external|internal|private'
                r'|view|pure|payable|virtual|override'
                r'|returns\s*\([^)]*\)'
                r'|\w+(?:\([^)]*\))?'   # modifiers with args
                r'|\s)*)',
                body, re.DOTALL):
            fn_name = m.group(1)
            params = m.group(2).strip()
            modifiers_text = m.group(3) or ""

            # Determine visibility (default: public in Solidity)
            vis = "public"
            for v in ("external", "internal", "private", "public"):
                if re.search(rf'\b{v}\b', modifiers_text):
                    vis = v
                    break

            fns.append({
                "name": fn_name,
                "params": params,
                "visibility": vis,
                "is_payable": bool(
                    re.search(r'\bpayable\b', modifiers_text)),
            })
        return fns

    def _extract_state_vars(self, body: str,
                            ) -> List[Dict[str, Any]]:
        """Extract state variable declarations.

        Matches standard Solidity types and contract/interface types
        used as state variables.
        """
        results: List[Dict[str, Any]] = []
        # Standard types
        for m in re.finditer(
                r'^\s*(mapping\s*\([^)]+\)|address|uint\d*|int\d*'
                r'|bool|string|bytes\d*)'
                r'\s+(?:public\s+|private\s+|internal\s+)?(\w+)',
                body, re.MULTILINE):
            results.append({
                "type": m.group(1).split("(")[0].strip(),
                "name": m.group(2),
            })
        # Contract/interface type state variables:
        # e.g., IERC20 public token;
        for m in re.finditer(
                r'^\s*(I?[A-Z]\w+)\s+'
                r'(?:public\s+|private\s+|internal\s+)?'
                r'(\w+)\s*;',
                body, re.MULTILINE):
            type_name = m.group(1)
            # Skip if it looks like a function (has parentheses later)
            # or is a known keyword
            if type_name in ("function", "event", "modifier",
                             "constructor", "contract", "library",
                             "interface", "struct", "enum",
                             "mapping", "returns"):
                continue
            results.append({
                "type": type_name,
                "name": m.group(2),
            })
        return results

    def _extract_ext_calls(self, body: str,
                           ) -> List[Dict[str, str]]:
        """Extract external call sites.

        Two categories:
          1. Low-level calls: addr.call{value:...}, addr.send(...),
             addr.transfer(...), addr.delegatecall(...)
          2. Typed interface calls: IPool(addr).deposit(...)
        """
        calls: List[Dict[str, str]] = []
        # Low-level calls
        for m in re.finditer(
                r'(\w+)\.(call|send|transfer|delegatecall)\b',
                body):
            calls.append({
                "target_var": m.group(1),
                "method": m.group(2),
            })
        # Typed interface calls: TypeName(expr).methodName(...)
        # Requires TypeName to start with uppercase (interface/contract)
        for m in re.finditer(
                r'([A-Z]\w+)\s*\([^)]*\)\s*\.\s*(\w+)\s*\(',
                body):
            type_name = m.group(1)
            method_name = m.group(2)
            # Skip abi.encode/decode patterns
            if type_name.lower() in ("abi", "type", "super"):
                continue
            calls.append({
                "interface": type_name,
                "method": method_name,
            })
        return calls

    # ================================================================
    # Interaction detectors
    # ================================================================

    def _detect_inheritance(self, g: InteractionGraph,
                            names: Set[str]):
        """INHERITANCE edges: contract A inherits from contract B."""
        for cn, ci in g.contracts.items():
            for parent in ci.inherits_from:
                if parent in names and parent != cn:
                    g.interactions.append(Interaction(
                        cn, parent, InteractionType.INHERITANCE))

    def _detect_external_calls(self, g: InteractionGraph,
                               names: Set[str]):
        """EXTERNAL_CALL edges: contract A calls contract B.

        Matching strategy:
          1. For low-level calls (.call, .send, .transfer), check if
             the target variable name exactly matches (case-insensitive)
             a known contract name, or if the variable is typed as
             that contract.
          2. For typed interface calls (IPool(...).method()), strip
             the I prefix from the interface name and match.

        The old implementation used substring matching (e.g., "Token"
        matched "stakingTokenBalance"), producing false edges.
        """
        # Pre-compute lowercase name lookup
        name_lower: Dict[str, str] = {n.lower(): n for n in names}

        for cn, ci in g.contracts.items():
            for call in ci.external_calls:
                target = None

                # Strategy 1: exact match on target variable name
                tv = call.get("target_var", "")
                if tv:
                    tv_lower = tv.lower()
                    if tv_lower in name_lower and \
                            name_lower[tv_lower] != cn:
                        target = name_lower[tv_lower]

                # Strategy 2: match typed interface call
                if target is None:
                    iface = call.get("interface", "")
                    if iface:
                        # Try exact match first
                        iface_lower = iface.lower()
                        if iface_lower in name_lower and \
                                name_lower[iface_lower] != cn:
                            target = name_lower[iface_lower]
                        else:
                            # Strip I prefix: IPool -> Pool
                            bare = _strip_interface_prefix(iface)
                            bare_lower = bare.lower()
                            if bare_lower in name_lower and \
                                    name_lower[bare_lower] != cn:
                                target = name_lower[bare_lower]

                # Strategy 3: check if source declares a state
                # variable typed as another contract
                if target is None and tv:
                    for other in names:
                        if other == cn:
                            continue
                        # Look for declaration: OtherContract tv;
                        # or OtherContract public tv;
                        decl_pat = (
                            rf'\b{re.escape(other)}\s+'
                            rf'(?:public\s+|private\s+|internal\s+)?'
                            rf'{re.escape(tv)}\b')
                        if re.search(decl_pat, ci.source_code):
                            target = other
                            break

                if target is not None:
                    g.interactions.append(Interaction(
                        cn, target,
                        InteractionType.EXTERNAL_CALL,
                        {"method": call.get("method", ""),
                         "via": tv or call.get("interface", "")}))

    def _detect_interface_deps(self, g: InteractionGraph,
                               names: Set[str]):
        """INTERFACE_DEP edges: contract A references interface IB
        that corresponds to contract B.

        Strips exactly one I prefix (not all leading I characters).
        """
        name_lower = {n.lower(): n for n in names}

        for cn, ci in g.contracts.items():
            for iface in ci.interfaces_used:
                bare = _strip_interface_prefix(iface)
                bare_lower = bare.lower()
                if bare_lower in name_lower:
                    target = name_lower[bare_lower]
                    if target != cn:
                        g.interactions.append(Interaction(
                            cn, target,
                            InteractionType.INTERFACE_DEP,
                            {"interface": iface}))

    def _detect_state_deps(self, g: InteractionGraph,
                           names: Set[str]):
        """STATE_DEPENDENCY edges: two contracts access the same
        named state variable.

        Uses word-boundary matching and skips very short variable
        names (length <= 2) to avoid spurious matches like "to"
        matching everywhere in source code.
        """
        # Collect per-variable access sets
        writers: Dict[str, Set[str]] = defaultdict(set)
        readers: Dict[str, Set[str]] = defaultdict(set)

        for cn, ci in g.contracts.items():
            for sv in ci.state_variables:
                vn = sv["name"]
                if len(vn) <= 2:
                    continue  # skip "x", "to", "id", etc.
                # Word-boundary matching to avoid substrings
                escaped = re.escape(vn)
                if re.search(rf'\b{escaped}\b\s*=',
                             ci.source_code):
                    writers[vn].add(cn)
                if re.search(rf'\b{escaped}\b',
                             ci.source_code):
                    readers[vn].add(cn)

        for vn, ws in writers.items():
            all_accessors = ws | readers.get(vn, set())
            if len(all_accessors) > 1:
                accessors = sorted(all_accessors)
                for i in range(len(accessors)):
                    for j in range(i + 1, len(accessors)):
                        g.interactions.append(Interaction(
                            accessors[i], accessors[j],
                            InteractionType.STATE_DEPENDENCY,
                            {"variable": vn}))

    def _detect_event_deps(self, g: InteractionGraph,
                           names: Set[str]):
        """EVENT_DEPENDENCY edges: contract A emits event E,
        contract B references event E (listener/reactor pattern).

        Uses word-boundary matching for event name lookups.
        """
        emitters: Dict[str, Set[str]] = defaultdict(set)
        for cn, ci in g.contracts.items():
            for ev in ci.events:
                escaped = re.escape(ev)
                if re.search(rf'\bemit\s+{escaped}\b',
                             ci.source_code):
                    emitters[ev].add(cn)

        for cn, ci in g.contracts.items():
            for ev, emitter_set in emitters.items():
                if cn in emitter_set:
                    continue  # skip self
                escaped = re.escape(ev)
                if re.search(rf'\b{escaped}\b', ci.source_code):
                    for em in emitter_set:
                        g.interactions.append(Interaction(
                            em, cn,
                            InteractionType.EVENT_DEPENDENCY,
                            {"event": ev}))

    def _detect_proxy_patterns(self, g: InteractionGraph,
                               names: Set[str]):
        """PROXY_DELEGATE edges: proxy contract A delegatecalls
        implementation contract B.

        Heuristic (SS3.2): pairs each proxy with the non-proxy,
        non-library contract that has the largest function set,
        under the assumption that the implementation contract
        exposes the full interface the proxy forwards.
        """
        for cn, ci in g.contracts.items():
            if not ci.is_proxy:
                continue
            # Find best implementation candidate
            best_name: Optional[str] = None
            best_fn_count = 0
            for other in names:
                if other == cn:
                    continue
                oci = g.contracts[other]
                if oci.is_proxy or oci.is_library:
                    continue
                if len(oci.functions) > best_fn_count:
                    best_fn_count = len(oci.functions)
                    best_name = other
            if best_name is not None and best_fn_count > 0:
                g.interactions.append(Interaction(
                    cn, best_name,
                    InteractionType.PROXY_DELEGATE,
                    {"pattern": "upgradeable"}))


# ═══════════════════════════════════════════════════════════════════════
# Cross-contract ground truth extraction
# ═══════════════════════════════════════════════════════════════════════

class CrossContractGroundTruth:
    """Extract cross-contract vulnerability indicators from DAppSCAN.

    IMPORTANT METHODOLOGICAL NOTE (SS4.1, SS6.4):
    DAppSCAN annotations are per-file SWC labels, NOT explicit
    cross-contract labels.  Two heuristics approximate cross-contract
    ground truth:

    (1) Multi-file indicator: vulnerabilities annotated across
        multiple distinct Solidity files, suggesting the vulnerability
        context spans contracts.
    (2) Interaction-prone SWC indicator: the SWC type commonly
        involves cross-contract behaviour.

    Confidence levels:
      "strong":   multi-file vulnerability + interaction-prone SWC
      "moderate": multi-file vulnerability only
      "weak":     interaction-prone SWC in single file, multi-file DApp

    These are PROXY indicators, not confirmed cross-contract labels.
    The paper states this in Threats to Validity (SS6.4).
    """

    INTERACTION_PRONE_SWCS: Dict[int, str] = {
        107: "reentrancy",
        104: "unchecked_call",
        105: "access_control",
        112: "delegatecall",
        114: "tod",
    }

    @staticmethod
    def extract(dapp_vulns: List[Dict[str, Any]],
                dapp_files: Dict[str, str],
                ) -> List[Dict[str, Any]]:
        """Identify vulnerability indicators that suggest
        cross-contract scope.

        Returns a list of indicator dicts, each with a 'confidence'
        field ("strong", "moderate", or "weak").
        """
        if not dapp_vulns:
            return []

        is_multi_file = len(dapp_files) > 1

        # Group vulnerabilities by file
        file_vulns: Dict[str, List[Dict]] = defaultdict(list)
        for v in dapp_vulns:
            fp = v.get("file", "")
            file_vulns[fp].append(v)

        vuln_files = [f for f, vs in file_vulns.items() if vs]
        has_multi_file_vulns = len(vuln_files) > 1

        indicators: List[Dict[str, Any]] = []

        # Heuristic 1: vulnerabilities span multiple files
        if has_multi_file_vulns:
            indicators.append({
                "type": "multi_file_vulnerability_indicator",
                "confidence": "moderate",
                "files": vuln_files,
                "n_files": len(vuln_files),
                "total_vulns": sum(
                    len(vs) for vs in file_vulns.values()),
                "note": "Vulnerabilities annotated across "
                        "multiple files",
            })

        # Heuristic 2: interaction-prone SWC types
        ip_swcs = CrossContractGroundTruth.INTERACTION_PRONE_SWCS
        for v in dapp_vulns:
            sid = v.get("swc_id", -1)
            if sid not in ip_swcs:
                continue

            if has_multi_file_vulns:
                conf = "strong"
            elif is_multi_file:
                conf = "weak"
            else:
                continue  # single-file DApp: not meaningful

            indicators.append({
                "type": f"interaction_prone_SWC_{sid}",
                "swc_name": ip_swcs[sid],
                "confidence": conf,
                "swc_id": sid,
                "file": v.get("file", ""),
                "function": v.get("function", ""),
                "note": ("Multi-file DApp with interaction-prone SWC"
                         if conf == "strong" else
                         "Multi-file DApp with interaction-prone SWC "
                         "(single-file vuln annotation)"),
            })

        return indicators


# ═══════════════════════════════════════════════════════════════════════
# Module-level helpers
# ═══════════════════════════════════════════════════════════════════════

def _strip_interface_prefix(name: str) -> str:
    """Strip exactly one 'I' prefix from an interface name.

    IPool       -> Pool
    IERC20      -> ERC20
    IIterableMap -> IterableMap  (not "terableMap" like lstrip("I"))
    Pool        -> Pool         (no prefix to strip)
    I           -> I            (single char, nothing to strip)
    """
    if (len(name) > 1
            and name[0] == "I"
            and name[1].isupper()):
        return name[1:]
    return name
