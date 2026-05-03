"""
underwriter_agent/judges.py
----------------------------
LLM-as-Judge validation subgraph with parallel fan-out.

Architecture (native LangGraph fan-out):
  fan_out → [factuality_judge, completeness_judge, consistency_judge] → merge_verdicts

The three judge nodes are siblings connected from a single fan_out node.
LangGraph's superstep model executes all siblings concurrently in the same
step — this is true parallelism, not a sequential for-loop.

State uses a list[JudgeResult] with operator.add reducer so that each judge's
partial write merges deterministically into the merge node.
"""

from __future__ import annotations

import json
import logging
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# ── Judge subgraph state ──────────────────────────────────────────────────────

class JudgeResult(TypedDict):
    judge: str
    passed: bool
    issues: list[str]


class JudgeSubgraphState(TypedDict):
    draft_response: str
    source_chunks: list[str]
    messages: list[BaseMessage]
    # reducer: list accumulates results from all three parallel judges
    judge_results: Annotated[list[JudgeResult], operator.add]
    judge_verdict: dict | None


# ── LLM helper ───────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _judge_llm_call(system_prompt: str, user_content: str) -> dict[str, Any]:
    from underwriter_agent.llm import get_llm

    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Judge 1: Factuality ───────────────────────────────────────────────────────

FACTUALITY_PROMPT = """\
You are a factuality judge for a life insurance underwriting AI.
Your job: verify every factual claim in the DRAFT RESPONSE against the SOURCE CHUNKS.
A claim FAILS if it cannot be found in the source chunks or contradicts them.

Return ONLY valid JSON:
{"pass": true/false, "issues": ["issue 1", "issue 2"]}
issues is an empty list if pass is true.
"""


def factuality_judge(state: JudgeSubgraphState) -> dict:
    draft = state["draft_response"]
    chunks = state["source_chunks"]

    if not chunks:
        # No retrieval was done (e.g. clarification question) — trivially pass
        return {"judge_results": [{"judge": "factuality", "passed": True, "issues": []}]}

    source_text = "\n\n---\n\n".join(chunks[:6])  # cap to avoid token overflow
    user_content = f"DRAFT RESPONSE:\n{draft}\n\nSOURCE CHUNKS:\n{source_text}"

    try:
        result = _judge_llm_call(FACTUALITY_PROMPT, user_content)
        passed = result.get("pass", False)
        issues = result.get("issues", [])
    except Exception as exc:
        logger.warning("Factuality judge LLM error: %s", exc)
        passed = True  # fail-open on judge error
        issues = []

    logger.info("Factuality judge: pass=%s issues=%s", passed, issues)
    return {"judge_results": [{"judge": "factuality", "passed": passed, "issues": issues}]}


# ── Judge 2: Completeness ────────────────────────────────────────────────────

COMPLETENESS_PROMPT = """\
You are a completeness judge for a life insurance underwriting AI.
Your job: check whether the DRAFT RESPONSE covers all materially relevant disclosures
present in the SOURCE CHUNKS that are relevant to the QUESTION CONTEXT.

Return ONLY valid JSON:
{"pass": true/false, "issues": ["missing item 1"]}
issues is an empty list if pass is true.
"""


def completeness_judge(state: JudgeSubgraphState) -> dict:
    draft = state["draft_response"]
    chunks = state["source_chunks"]
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "human":
            last_user_msg = msg.content
            break

    if not chunks:
        return {"judge_results": [{"judge": "completeness", "passed": True, "issues": []}]}

    source_text = "\n\n---\n\n".join(chunks[:6])
    user_content = (
        f"QUESTION CONTEXT: {last_user_msg}\n\n"
        f"DRAFT RESPONSE:\n{draft}\n\n"
        f"SOURCE CHUNKS:\n{source_text}"
    )

    try:
        result = _judge_llm_call(COMPLETENESS_PROMPT, user_content)
        passed = result.get("pass", False)
        issues = result.get("issues", [])
    except Exception as exc:
        logger.warning("Completeness judge LLM error: %s", exc)
        passed = True
        issues = []

    logger.info("Completeness judge: pass=%s issues=%s", passed, issues)
    return {"judge_results": [{"judge": "completeness", "passed": passed, "issues": issues}]}


# ── Judge 3: Consistency ──────────────────────────────────────────────────────

CONSISTENCY_PROMPT = """\
You are a consistency judge for a life insurance underwriting AI.
Your job: check ONLY for genuine contradictions:
  1. Does the DRAFT RESPONSE contradict itself internally?
  2. Does it state a DIFFERENT fact from something already confirmed in CONVERSATION HISTORY?

IMPORTANT RULES:
- New information not previously mentioned is NOT a contradiction — it is expected
- Only flag if the same fact is stated differently (e.g. said $500k before, now says $300k)
- If conversation history is short or only contains the user question, pass=true
- When in doubt, pass=true

Return ONLY valid JSON:
{"pass": true/false, "issues": ["only actual contradictions here"]}
"""


def consistency_judge(state: JudgeSubgraphState) -> dict:
    draft = state["draft_response"]
    # Summarise last N turns for consistency checking (avoid huge context)
    history_text = "\n".join(
        f"{msg.type.upper()}: {msg.content[:300]}"
        for msg in state["messages"][-10:]
    )

    user_content = (
        f"CONVERSATION HISTORY (last 10 messages):\n{history_text}\n\n"
        f"DRAFT RESPONSE:\n{draft}"
    )

    try:
        result = _judge_llm_call(CONSISTENCY_PROMPT, user_content)
        passed = result.get("pass", False)
        issues = result.get("issues", [])
    except Exception as exc:
        logger.warning("Consistency judge LLM error: %s", exc)
        passed = True
        issues = []

    logger.info("Consistency judge: pass=%s issues=%s", passed, issues)
    return {"judge_results": [{"judge": "consistency", "passed": passed, "issues": issues}]}


# ── Fan-out node ──────────────────────────────────────────────────────────────

def fan_out(state: JudgeSubgraphState) -> dict:
    """Entry point; resets judge_results so partial writes accumulate cleanly."""
    return {"judge_results": []}


# ── Merge node ────────────────────────────────────────────────────────────────

def merge_verdicts(state: JudgeSubgraphState) -> dict:
    """
    Merge the three JudgeResults into a single verdict dict written back to
    the parent AgentState.judge_verdict field.
    """
    results = state["judge_results"]
    by_name = {r["judge"]: r for r in results}

    factuality = by_name.get("factuality", {"passed": True, "issues": []})
    completeness = by_name.get("completeness", {"passed": True, "issues": []})
    consistency = by_name.get("consistency", {"passed": True, "issues": []})

    all_passed = factuality["passed"] and completeness["passed"] and consistency["passed"]
    all_issues = (
        factuality["issues"] + completeness["issues"] + consistency["issues"]
    )
    n_judges = 3
    n_passed = sum([factuality["passed"], completeness["passed"], consistency["passed"]])
    confidence = round(n_passed / n_judges, 2)

    verdict = {
        "overall_pass": all_passed,
        "factuality": {"pass": factuality["passed"], "issues": factuality["issues"]},
        "completeness": {"pass": completeness["passed"], "issues": completeness["issues"]},
        "consistency": {"pass": consistency["passed"], "issues": consistency["issues"]},
        "confidence_score": confidence,
    }
    logger.info("Judge verdict: overall_pass=%s confidence=%s", all_passed, confidence)
    return {"judge_verdict": verdict}


# ── Build and compile the subgraph ────────────────────────────────────────────

def build_judge_subgraph():
    """
    Compile and return the judge subgraph.

    Fan-out pattern:
      fan_out → factuality_judge  ┐
              → completeness_judge├→ merge_verdicts → END
              → consistency_judge ┘

    All three judge nodes receive edges from fan_out simultaneously.
    LangGraph executes them in the same superstep (concurrent).
    The operator.add reducer on judge_results merges partial writes deterministically.
    """
    g = StateGraph(JudgeSubgraphState)

    g.add_node("fan_out", fan_out)
    g.add_node("factuality_judge", factuality_judge)
    g.add_node("completeness_judge", completeness_judge)
    g.add_node("consistency_judge", consistency_judge)
    g.add_node("merge_verdicts", merge_verdicts)

    g.set_entry_point("fan_out")

    # True parallel fan-out: three edges from one node
    g.add_edge("fan_out", "factuality_judge")
    g.add_edge("fan_out", "completeness_judge")
    g.add_edge("fan_out", "consistency_judge")

    # All judges fan in to merge
    g.add_edge("factuality_judge", "merge_verdicts")
    g.add_edge("completeness_judge", "merge_verdicts")
    g.add_edge("consistency_judge", "merge_verdicts")

    g.add_edge("merge_verdicts", END)

    return g.compile()
