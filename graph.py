"""
underwriter_agent/graph.py
---------------------------
Main agent StateGraph.

Per-turn flow:
  agent → (tool calls loop) → agent → draft_response → judge_subgraph
       → revise-or-emit (conditional) → emit → END

Checkpointer:
  MemorySaver (in-process, for this assessment).
  Production: SqliteSaver (single-node) or PostgresSaver (multi-node/HA).
  Thread ID is keyed on record_id so each application has its own isolated
  conversation thread.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from underwriter_agent.tools import (
    document_search,
    underwriting_analysis,
    summary_generator,
    reset_turn_chunks,
    get_turn_chunks,
    set_retriever,
    set_full_pdf_text,
)

logger = logging.getLogger(__name__)

TOOLS = [document_search, underwriting_analysis, summary_generator]

MAX_REVISIONS = 1  # revise at most once before emitting with a warning

SYSTEM_PROMPT = """\
You are an expert life insurance underwriting assistant for NobleOak Life Limited (Australia).
You have access to three tools:
  • document_search(query) – search the loaded PDF application for relevant passages
  • underwriting_analysis(topic) – apply underwriting heuristics to a specific topic
  • summary_generator() – produce the full structured underwriting summary

Guidelines:
  - NEVER hallucinate or infer facts not found in the application.
  - If a fact is absent, say so explicitly.
  - Always use document_search or underwriting_analysis before answering factual questions.
  - For the final underwriting summary, always call summary_generator().
  - Be concise and professional.
"""


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    record_id: str
    draft_response: str | None
    # source_chunks accumulated this turn — judges read from here
    source_chunks: Annotated[list[str], operator.add]
    judge_verdict: dict | None
    revision_count: int


# ── Agent node ────────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """
    Core reasoning node. Binds the LLM to the three tools.
    Uses create_react_agent style: bind tools to LLM, inject system prompt.

    Why hand-rolled rather than create_react_agent:
      create_react_agent wraps the graph in a fixed schema that makes it harder
      to inject the judge subgraph as an intermediate step. Hand-rolling gives
      explicit control over when draft_response is set vs when tool calls loop.
    """
    from underwriter_agent.llm import get_llm

    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    # Reset source chunks at the start of each new user turn
    # (revision turns carry over existing chunks intentionally)
    if state.get("revision_count", 0) == 0:
        reset_turn_chunks()

    # Prepend system prompt if this is the first message
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response: AIMessage = llm_with_tools.invoke(messages)

    update: dict = {"messages": [response]}

    # If no tool calls → this is a final reply; capture as draft
    if not response.tool_calls:
        update["draft_response"] = response.content
        update["source_chunks"] = get_turn_chunks()  # snapshot for judges
        logger.info(
            "agent_node: final reply produced (len=%d)", len(response.content)
        )
    else:
        logger.info(
            "agent_node: %d tool call(s) requested", len(response.tool_calls)
        )

    return update


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_agent(state: AgentState) -> str:
    """Route to tools if last AI message has tool calls, else to judge subgraph."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "judges"


def route_after_judges(state: AgentState) -> str:
    """
    Revise-or-emit control flow:
      • overall_pass → emit
      • failed + revision_count < MAX_REVISIONS → revise (loop to agent)
      • failed + revision_count >= MAX_REVISIONS → emit with warning
    """
    verdict = state.get("judge_verdict") or {}
    overall_pass = verdict.get("overall_pass", True)
    revision_count = state.get("revision_count", 0)

    if overall_pass:
        logger.info("route_after_judges: PASS → emit")
        return "emit"

    if revision_count < MAX_REVISIONS:
        logger.info(
            "route_after_judges: FAIL revision_count=%d → revise", revision_count
        )
        return "revise"

    logger.info(
        "route_after_judges: FAIL revision_count=%d >= MAX → emit with warning",
        revision_count,
    )
    return "emit"


# ── Revise node ───────────────────────────────────────────────────────────────

def revise_node(state: AgentState) -> dict:
    """
    Append judge feedback as a system message and increment revision_count,
    then return to agent for one more attempt.
    """
    verdict = state.get("judge_verdict", {})
    all_issues: list[str] = []
    for key in ("factuality", "completeness", "consistency"):
        section = verdict.get(key, {})
        all_issues.extend(section.get("issues", []))

    feedback_text = (
        "Revise your previous answer — here is the judge feedback:\n"
        + "\n".join(f"  • {issue}" for issue in all_issues)
    )
    feedback_msg = SystemMessage(content=feedback_text)
    logger.info("revise_node: injecting feedback: %s", all_issues)
    return {
        "messages": [feedback_msg],
        "revision_count": state.get("revision_count", 0) + 1,
        "source_chunks": [],  # reset so chunks don't double-accumulate
    }


# ── Emit node ─────────────────────────────────────────────────────────────────

def emit_node(state: AgentState) -> dict:
    """
    Final output node. Optionally prepends a warning block if judges flagged
    issues but we've exhausted the revision budget.
    """
    verdict = state.get("judge_verdict", {})
    draft = state.get("draft_response", "")
    overall_pass = verdict.get("overall_pass", True)
    revision_count = state.get("revision_count", 0)

    if not overall_pass and revision_count >= MAX_REVISIONS:
        # Collect all judge issues for the warning block
        all_issues: list[str] = []
        for key in ("factuality", "completeness", "consistency"):
            section = verdict.get(key, {})
            all_issues.extend(section.get("issues", []))

        warning = (
            "\n\n■ Judge flagged: The following concerns were identified but could not be "
            "fully resolved:\n"
            + "\n".join(f"  • {issue}" for issue in all_issues)
        )
        final_text = draft + warning
    else:
        final_text = draft

    # Replace the draft AI message with a clean final one
    final_msg = AIMessage(content=final_text)
    logger.info("emit_node: emitting final response (len=%d)", len(final_text))
    return {
        "messages": [final_msg],
        "draft_response": None,
        "source_chunks": [],
        "judge_verdict": None,
        "revision_count": 0,
    }


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(retriever=None, full_pdf_text: str = "", checkpointer=None):
    """
    Build and compile the main StateGraph.

    Args:
        retriever:       FAISS retriever from build_retriever()
        full_pdf_text:   Raw PDF text for summary_generator
        checkpointer:    LangGraph checkpointer (default: MemorySaver)
    """
    if retriever is not None:
        set_retriever(retriever)
    if full_pdf_text:
        set_full_pdf_text(full_pdf_text)

    from underwriter_agent.judges import build_judge_subgraph

    judge_subgraph = build_judge_subgraph()
    tool_node = ToolNode(TOOLS)

    g = StateGraph(AgentState)

    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_node("judges", judge_subgraph)
    g.add_node("revise", revise_node)
    g.add_node("emit", emit_node)

    g.set_entry_point("agent")

    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "judges": "judges"},
    )
    g.add_edge("tools", "agent")

    # judges subgraph → merge → back to parent via judge_verdict in state
    # The subgraph outputs judge_verdict; the parent conditional edge then routes
    g.add_conditional_edges(
        "judges",
        route_after_judges,
        {"emit": "emit", "revise": "revise"},
    )
    g.add_edge("revise", "agent")
    g.add_edge("emit", END)

    cp = checkpointer or MemorySaver()
    compiled = g.compile(checkpointer=cp)
    logger.info("Graph compiled successfully")
    return compiled
