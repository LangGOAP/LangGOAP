r"""Deep research agent — Tier 3 tutorial translating deepagents' example.

Adapts the orchestrator-plus-subagent research loop from
``research/repos/deepagents/examples/deep_research`` into a LangGoap plan
driven by the Layer A on-ramp :func:`langgoap.integrations.create_goap_agent`.
The tutorial spotlights five features at once:

1. **Layer A ``create_goap_agent``** — the "most prebuilt" low-code
   on-ramp.  The tutorial wraps ordinary LangChain ``@tool`` functions,
   passes explicit preconditions/effects, and gets back a compiled
   LangGraph graph with no manual ``ActionSpec`` plumbing.
2. **``PlanningTracer`` observability** — a :class:`LoggingTracer` and
   a :class:`MultiTracer` capture every plan/replan/goal-achieved event
   so the notebook can show the full orchestrator loop in human-readable
   form.
3. **``StoreExecutionHistory`` persistence** — an
   :class:`langgraph.store.memory.InMemoryStore` records every run as
   an :class:`ExecutionRecord`; the tutorial queries the history across
   multiple invocations to demonstrate the reverse-index pattern.
4. **Multi-step replanning** — the cheap ``search_broad_corpus`` tool
   raises :class:`RateLimitError` against the default rate-limited
   workspace.  The executor blacklists the action and the planner's
   fallback routes through the more expensive ``search_deep_corpus``
   on the next pass.
5. **NL goal intake** — a :class:`langgoap.testing.FakeStructuredModel`
   drives :meth:`GoapGraph.invoke_nl` end-to-end, mirroring the
   pattern already used in the Tier 2 notebooks.

Everything is hermetic: no shell commands, no real web traffic, no
external dependencies.  The :class:`ResearchWorkspace` mutable object
in this module is the only source of truth for corpus, findings, and
the final synthesized report — the LangGoap layer tracks phase flags
(``request_saved``, ``topics_planned``, etc.) while the workspace
holds the actual research data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool, tool

from langgoap import GoalSpec

from .data.deep_research_instance import (
    CORPUS,
    MIN_REPORT_CITATIONS,
    RESEARCH_REQUEST,
    ResearchDocument,
)

# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class RateLimitError(RuntimeError):
    """Raised by the broad search tool when the corpus is rate-limited.

    Subclassing :class:`RuntimeError` keeps the executor's exception
    handling path identical to the one exercised by the vulnerability
    scanner tutorial — the executor catches any ``Exception`` raised
    from ``execute`` and routes the action through the blacklist
    branch.
    """


class InsufficientEvidenceError(RuntimeError):
    """Raised by the synthesize tool when too few findings were gathered.

    The validator branch of the vulnerability scanner tutorial covers
    the post-state rejection path; here we exercise the *pre-state*
    exception path so the tutorial's replan scenario is deterministic
    without also needing an ``effect_validator``.
    """


# ---------------------------------------------------------------------------
# Research workspace — mutable state holder for the research run
# ---------------------------------------------------------------------------


@dataclass
class ResearchWorkspace:
    """Hermetic stand-in for deepagents' filesystem + tavily pipeline.

    The workspace owns the corpus and the intermediate research
    artifacts (request text, topic decomposition, gathered findings,
    final report).  Tools close over a workspace instance, mutate it
    during ``execute``, and the GOAP layer tracks only the phase flags
    that drive planning.

    Set ``rate_limit_active`` to ``False`` to let the cheap broad
    search succeed on its first attempt — used by tests that want to
    exercise the happy-path planner without triggering the blacklist
    branch.
    """

    corpus: tuple[ResearchDocument, ...] = CORPUS
    rate_limit_active: bool = True
    # Filled in as the research loop progresses.
    research_request: str | None = None
    topics: list[str] = field(default_factory=list)
    findings: list[ResearchDocument] = field(default_factory=list)
    report: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Core research operations
    # ------------------------------------------------------------------

    def save_request(self, query: str) -> None:
        """Persist the user's research question to the workspace."""
        self.research_request = query

    def decompose(self) -> list[str]:
        """Extract topic tags present in both the request and the corpus.

        Mirrors deepagents' ``write_todos`` step: break the free-text
        research question into a small set of focused research tasks.
        The decomposition is deterministic — the corpus tag vocabulary
        is scanned in insertion order so the tutorial's expected-topic
        list is stable.
        """
        if self.research_request is None:
            raise RuntimeError(
                "decompose() called before save_request(); no request saved"
            )
        # Hyphens in phrases like "retrieval-augmented" become spaces so
        # the tag vocabulary (``retrieval augmented generation``) can
        # match them after the ``tag.replace("_", " ")`` step.
        normalized = self.research_request.lower().replace("-", " ")
        seen: list[str] = []
        for doc in self.corpus:
            for tag in doc.tags:
                if tag in seen:
                    continue
                keyword = tag.replace("_", " ")
                if keyword in normalized:
                    seen.append(tag)
        self.topics = seen
        return list(seen)

    def search_broad(self) -> list[ResearchDocument]:
        """Cheap corpus scan — returns one document per topic, or raises.

        When ``rate_limit_active`` is ``True``, raises
        :class:`RateLimitError` without gathering any findings.  This
        lets the tutorial exercise the executor's exception path and
        the planner's blacklist + fallback behavior.
        """
        if self.rate_limit_active:
            raise RateLimitError(
                "rate limit exceeded on broad corpus search"
            )
        findings: list[ResearchDocument] = []
        seen: set[str] = set()
        for topic in self.topics:
            for doc in self.corpus:
                if topic in doc.tags and doc.id not in seen:
                    findings.append(doc)
                    seen.add(doc.id)
                    break
        self.findings = findings
        return list(findings)

    def search_deep(self) -> list[ResearchDocument]:
        """Thorough corpus scan — every document matching any topic.

        Always succeeds regardless of ``rate_limit_active``; the cost
        is reflected in the ``ActionSpec`` cost/resources rather than
        in runtime behavior.
        """
        findings: list[ResearchDocument] = []
        seen: set[str] = set()
        for doc in self.corpus:
            if any(t in self.topics for t in doc.tags) and doc.id not in seen:
                findings.append(doc)
                seen.add(doc.id)
        self.findings = findings
        return list(findings)

    def synthesize(self) -> dict[str, Any]:
        """Build the final report from the gathered findings.

        Raises :class:`InsufficientEvidenceError` when too few unique
        citations were gathered — the same failure surface
        ``effect_validator`` would cover, expressed as an exception so
        it is caught by the executor's normal failure path.
        """
        unique_urls = {f.url for f in self.findings}
        if len(unique_urls) < MIN_REPORT_CITATIONS:
            raise InsufficientEvidenceError(
                f"only {len(unique_urls)} unique citations, "
                f"need {MIN_REPORT_CITATIONS}"
            )
        sections = []
        for finding in self.findings:
            sections.append(
                {
                    "title": finding.title,
                    "url": finding.url,
                    "snippet": finding.snippet,
                }
            )
        self.report = {
            "request": self.research_request,
            "topics": list(self.topics),
            "citations": [
                {"title": f.title, "url": f.url} for f in self.findings
            ],
            "sections": sections,
        }
        return dict(self.report)


# ---------------------------------------------------------------------------
# LangChain tool factories — every tool closes over a single workspace
# ---------------------------------------------------------------------------


def deep_research_tools(workspace: ResearchWorkspace) -> list[BaseTool]:
    """Return the five research tools bound to *workspace*.

    **Why every tool declares a ``query`` parameter it may not use.**
    Layer B's :func:`goapify_tool` wrapper binds tool arguments from
    the world-state dict by *name*: for each field declared in a
    tool's ``args_schema`` it reads the matching key from
    ``state["world_state"]`` and forwards it.  The planner never
    invents values and a missing key raises, so every research tool
    must declare ``query`` to receive the current request string from
    world state — even the four tools that ignore the value and read
    the previously-saved request off the workspace instead.

    The unused parameter is deleted with ``del query`` so linters and
    human readers can see the argument is intentionally discarded.
    The downside of this Layer A design constraint is that the tool
    signatures become coupled to the world-state key names; the
    upside is that Layer A never has to parse tool docstrings or
    guess at argument bindings.
    """

    @tool
    def save_research_request(query: str) -> str:
        """Persist the user's research question to the workspace."""
        workspace.save_request(query)
        return f"Saved research request: {query!r}"

    @tool
    def decompose_topics(query: str) -> str:
        """Break the research question into focused subtopics.

        ``query`` is forwarded from world state but not read here —
        the workspace already persists the saved request.  See the
        module-level ``deep_research_tools`` docstring for the
        Layer A binding rule that forces this parameter to exist.
        """
        del query  # parameter required by Layer A binding, not used
        topics = workspace.decompose()
        return f"Decomposed into {len(topics)} topics: {topics}"

    @tool
    def search_broad_corpus(query: str) -> str:
        """Cheap corpus scan — raises when the workspace is rate-limited."""
        del query  # parameter required by Layer A binding, not used
        findings = workspace.search_broad()
        return f"Broad search found {len(findings)} documents"

    @tool
    def search_deep_corpus(query: str) -> str:
        """Thorough corpus scan — always succeeds."""
        del query  # parameter required by Layer A binding, not used
        findings = workspace.search_deep()
        return f"Deep search found {len(findings)} documents"

    @tool
    def synthesize_report(query: str) -> str:
        """Consolidate the gathered findings into a final report."""
        del query  # parameter required by Layer A binding, not used
        report = workspace.synthesize()
        return f"Report synthesized with {len(report['citations'])} citations"

    return [
        save_research_request,
        decompose_topics,
        search_broad_corpus,
        search_deep_corpus,
        synthesize_report,
    ]


# ---------------------------------------------------------------------------
# Preconditions / effects / costs / start state / goal
# ---------------------------------------------------------------------------


def deep_research_preconditions() -> dict[str, dict[str, Any]]:
    """Phase-flag preconditions keyed by tool name (for Layer A)."""
    return {
        "save_research_request": {"request_saved": False},
        "decompose_topics": {
            "request_saved": True,
            "topics_planned": False,
        },
        "search_broad_corpus": {
            "topics_planned": True,
            "findings_gathered": False,
        },
        "search_deep_corpus": {
            "topics_planned": True,
            "findings_gathered": False,
        },
        "synthesize_report": {
            "findings_gathered": True,
            "report_written": False,
        },
    }


def deep_research_effects() -> dict[str, dict[str, Any]]:
    """Phase-flag effects keyed by tool name (for Layer A)."""
    return {
        "save_research_request": {"request_saved": True},
        "decompose_topics": {"topics_planned": True},
        "search_broad_corpus": {"findings_gathered": True},
        "search_deep_corpus": {"findings_gathered": True},
        "synthesize_report": {"report_written": True},
    }


def deep_research_costs() -> dict[str, float]:
    """Per-tool costs — cheap broad search preferred over deep search."""
    return {
        "save_research_request": 1.0,
        "decompose_topics": 1.0,
        "search_broad_corpus": 2.0,
        "search_deep_corpus": 5.0,
        "synthesize_report": 1.0,
    }


def deep_research_start(query: str = RESEARCH_REQUEST) -> dict[str, Any]:
    """Initial world state: nothing researched, no report yet.

    The ``query`` key is picked up by :func:`goapify_tool`'s state
    filter (each research tool declares a ``query`` argument), so the
    same string reaches every tool invocation in turn.  The default is
    the tutorial's canonical research question.
    """
    return {
        "query": query,
        "request_saved": False,
        "topics_planned": False,
        "findings_gathered": False,
        "report_written": False,
    }


def deep_research_goal() -> GoalSpec:
    """Goal: produce a written research report for the operator."""
    return GoalSpec(conditions={"report_written": True})
