r"""Hermetic research corpus for the deep-research-agent tutorial.

Provenance
----------
Structurally adapted from
``research/repos/deepagents/examples/deep_research/research_agent`` which
drives a tool-calling research loop against ``tavily_search`` (a live web
API).  For a deterministic, hermetic tutorial we replace the network call
with an in-memory :class:`ResearchDocument` corpus.  Each document is
tagged with a small set of topics, and two hand-picked "research
questions" are shipped alongside the corpus with the exact findings,
topics, and synthesized report that the agent is expected to produce.

The tutorial mirrors deepagents' orchestrator loop:

1. ``save_research_request``   — analogue of ``write_file('/research_request.md')``
2. ``decompose_topics``        — analogue of ``write_todos``
3. ``search_broad_corpus`` /
   ``search_deep_corpus``      — analogues of ``tavily_search``
4. ``synthesize_report``       — analogue of ``write_file('/final_report.md')``

The two search variants let the tutorial show replanning: the cheaper
broad search raises :class:`RateLimitError` when the corpus is
rate-limited, so the planner's blacklist fallback routes through the
more expensive deep search on the second attempt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchDocument:
    """One entry in the simulated research corpus."""

    id: str
    title: str
    url: str
    snippet: str
    tags: tuple[str, ...]


# ---------------------------------------------------------------------------
# Corpus — 10 hand-picked AI/ML documents.  Real titles and arXiv URLs, the
# snippets are condensed for the tutorial.
# ---------------------------------------------------------------------------

CORPUS: tuple[ResearchDocument, ...] = (
    ResearchDocument(
        id="doc_1",
        title="Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        snippet=(
            "Introduces the Transformer architecture built entirely on "
            "self-attention, eschewing recurrence and convolutions."
        ),
        tags=("transformers", "attention", "deep_learning"),
    ),
    ResearchDocument(
        id="doc_2",
        title="The Illustrated Transformer",
        url="https://jalammar.github.io/illustrated-transformer/",
        snippet=(
            "A visual walkthrough of multi-head self-attention, positional "
            "encodings, and the encoder-decoder stack."
        ),
        tags=("transformers", "attention", "education"),
    ),
    ResearchDocument(
        id="doc_3",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP",
        url="https://arxiv.org/abs/2005.11401",
        snippet=(
            "RAG couples a parametric seq2seq generator with a non-parametric "
            "dense retriever over Wikipedia."
        ),
        tags=("retrieval_augmented_generation", "knowledge_base", "nlp"),
    ),
    ResearchDocument(
        id="doc_4",
        title="Dense Passage Retrieval for Open-Domain QA",
        url="https://arxiv.org/abs/2004.04906",
        snippet=(
            "Learns dense passage embeddings with a bi-encoder trained on "
            "question-answer pairs, outperforming BM25 on five QA datasets."
        ),
        tags=("retrieval_augmented_generation", "dense_retrieval", "nlp"),
    ),
    ResearchDocument(
        id="doc_5",
        title="ReAct: Synergizing Reasoning and Acting in Language Models",
        url="https://arxiv.org/abs/2210.03629",
        snippet=(
            "Interleaves chain-of-thought reasoning with tool calls, letting "
            "the model query the world between thoughts."
        ),
        tags=("agent_frameworks", "reasoning", "tool_use"),
    ),
    ResearchDocument(
        id="doc_6",
        title="LangGraph: Stateful, Multi-Actor LLM Applications",
        url="https://langchain-ai.github.io/langgraph/",
        snippet=(
            "A graph-based runtime for building stateful LLM applications "
            "with persistent checkpoints and durable control flow."
        ),
        tags=("agent_frameworks", "orchestration", "langchain"),
    ),
    ResearchDocument(
        id="doc_7",
        title="Toolformer: Language Models Can Teach Themselves to Use Tools",
        url="https://arxiv.org/abs/2302.04761",
        snippet=(
            "Self-supervised fine-tuning pipeline that teaches a model when "
            "to call a small set of APIs (calculator, search, translation)."
        ),
        tags=("agent_frameworks", "tool_use", "self_supervision"),
    ),
    ResearchDocument(
        id="doc_8",
        title="Goal-Oriented Action Planning for Games",
        url="https://alumni.media.mit.edu/~jorkin/goap.html",
        snippet=(
            "The original GOAP writeup from F.E.A.R., motivating plan-space "
            "search for NPCs over scripted finite-state machines."
        ),
        tags=("goap", "planning", "game_ai"),
    ),
    ResearchDocument(
        id="doc_9",
        title="Reinforcement Learning from Human Feedback",
        url="https://arxiv.org/abs/2203.02155",
        snippet=(
            "The InstructGPT paper — a three-stage pipeline: supervised "
            "fine-tuning, reward modeling, then PPO against the reward model."
        ),
        tags=("alignment", "rlhf", "instruction_tuning"),
    ),
    ResearchDocument(
        id="doc_10",
        title="Constitutional AI: Harmlessness from AI Feedback",
        url="https://arxiv.org/abs/2212.08073",
        snippet=(
            "Replaces human red-teamers with an AI critic that scores "
            "responses against a written constitution."
        ),
        tags=("alignment", "self_critique", "rlhf"),
    ),
)


# ---------------------------------------------------------------------------
# Canonical research request exercised by the integration test and notebook.
# ---------------------------------------------------------------------------

RESEARCH_REQUEST: str = (
    "How do modern AI agent frameworks combine retrieval-augmented "
    "generation with transformers?"
)


# Topics the ``decompose_topics`` tool is expected to extract from the
# request.  Decomposition scans corpus documents in insertion order and
# records each new matching tag as it appears; the resulting order is
# stable and driven by the corpus, not by the request wording.
EXPECTED_TOPICS: tuple[str, ...] = (
    "transformers",  # first seen on doc_1
    "retrieval_augmented_generation",  # first seen on doc_3
    "agent_frameworks",  # first seen on doc_5
)


# Broad search pulls the first document per topic by corpus order.
EXPECTED_BROAD_FINDING_IDS: tuple[str, ...] = (
    "doc_1",  # first transformers match
    "doc_3",  # first retrieval_augmented_generation match
    "doc_5",  # first agent_frameworks match
)


# Deep search pulls every document whose tag set intersects any topic.
EXPECTED_DEEP_FINDING_IDS: tuple[str, ...] = (
    "doc_1",  # transformers
    "doc_2",  # transformers
    "doc_3",  # retrieval_augmented_generation
    "doc_4",  # retrieval_augmented_generation
    "doc_5",  # agent_frameworks
    "doc_6",  # agent_frameworks
    "doc_7",  # agent_frameworks
)


# The synthesized report must cite every deep-search finding — the
# validator rejects reports with fewer than ``MIN_REPORT_CITATIONS``
# unique URLs so the tutorial can drive the replan + error-recovery path.
MIN_REPORT_CITATIONS: int = 3
