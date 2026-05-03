"""
underwriter_agent/tools.py
---------------------------
The three @tool-decorated functions exposed to the agent.

  1. document_search   – RAG over FAISS
  2. underwriting_analysis – specialist LLM analysis of a topic
  3. summary_generator – full structured underwriting summary

A module-level retriever reference is set by build_graph() at startup
so tools share the same FAISS index without passing state around.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Set by build_graph() before first use
_retriever: VectorStoreRetriever | None = None
# Accumulates source chunks for the current turn (reset per turn by the agent node)
_current_turn_chunks: list[str] = []


def set_retriever(retriever: VectorStoreRetriever) -> None:
    global _retriever
    _retriever = retriever


def reset_turn_chunks() -> None:
    global _current_turn_chunks
    _current_turn_chunks = []


def get_turn_chunks() -> list[str]:
    return list(_current_turn_chunks)


# ── 1. document_search ────────────────────────────────────────────────────────

@tool
def document_search(query: str) -> str:
    """
    Search the loaded life insurance application PDF for passages relevant to the query.
    Returns the most relevant text chunks. Use this for any fact-finding about the applicant.
    """
    if _retriever is None:
        return "ERROR: Retriever not initialised. Call set_retriever() first."

    try:
        docs = _retriever.invoke(query)
    except Exception as exc:
        logger.exception("document_search failed")
        return f"Tool error: {exc}"

    if not docs:
        return (
            "I don't know — this information is not present in the application. "
            "No relevant sections were found for the query."
        )

    chunks = [d.page_content for d in docs]
    _current_turn_chunks.extend(chunks)

    result = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    logger.info("document_search: query=%r → %d chunks returned", query, len(chunks))
    return result


# ── 2. underwriting_analysis ─────────────────────────────────────────────────

UNDERWRITING_ANALYST_PROMPT = """\
You are a senior life insurance underwriter (Australia).
Below is the COMPLETE text of an insurance application.
Extract facts ONLY from this text — do not infer or assume anything not stated.

Produce a structured JSON summary with this EXACT structure:
{
  "applicant_header": {
    "name": "Mr John Smith",
    "dob": "21-08-1982", 
    "state": "NSW",
    "height_cm": 174,
    "weight_kg": 80,
    "bmi": 26.40
  },
  "sections": {
    "Applicant details": "Full name, DOB, gender, height, weight, BMI, state, residency status",
    "Applied for cover": "List each product with sum insured and premium",
    "Existing cover": "Any existing insurance with other insurers",
    "Modified Terms": "Any previous declines, loadings or exclusions",
    "Claims history": "Any past or intended claims",
    "Residency": "Visa type, living in Australia yes/no",
    "Occupation": "Job title, class, hours, second job if any",
    "Income": "Main job income, second job income, total",
    "Travel": "Planned overseas travel details including dates and destinations",
    "Recreation": "All sports, hobbies, aviation, diving disclosed",
    "Alcohol": "Weekly consumption, any counselling or treatment",
    "Drug Use": "Any drug use disclosed",
    "Smoking": "Smoker yes/no, how many per day",
    "BMI": "BMI value and category",
    "Medical History": "ALL medical conditions disclosed, with dates and status",
    "Family History": "Any family medical history disclosed",
    "GP Details": "GP name, address, how long a patient"
  },
  "red_flags": [
    "List every underwriting concern, inconsistency or missing info as a separate string"
  ],
  "disclosure_cross_check": [
    "List confirmations of items checked that were clear (bankruptcy, HIV, COVID etc)"
  ]
}

IMPORTANT: Replace the placeholder descriptions above with actual facts from the application.
Return ONLY valid JSON. No markdown, no explanation.
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_llm_for_analysis(topic: str, context: str) -> dict[str, Any]:
    from underwriter_agent.llm import get_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm()
    messages = [
        SystemMessage(content=UNDERWRITING_ANALYST_PROMPT),
        HumanMessage(
            content=f"TOPIC: {topic}\n\nAPPLICATION EXCERPT:\n{context}\n\nReturn JSON only."
        ),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


@tool
def underwriting_analysis(topic: str) -> dict:
    """
    Perform a structured underwriting analysis for a specific topic (e.g. 'medical history',
    'smoking', 'travel', 'hazardous activities', 'BMI', 'cancer history').
    Retrieves relevant application sections and applies underwriting heuristics.
    Returns findings, red_flags, risk_level, and follow_up_required.
    """
    if _retriever is None:
        return {"error": "Retriever not initialised."}

    try:
        docs = _retriever.invoke(topic)
    except Exception as exc:
        logger.exception("underwriting_analysis: retrieval failed")
        return {"error": f"Retrieval error: {exc}"}

    if not docs:
        return {
            "findings": f"No application content found for topic: {topic}",
            "red_flags": [],
            "risk_level": "low",
            "follow_up_required": [],
        }

    chunks = [d.page_content for d in docs]
    _current_turn_chunks.extend(chunks)
    context = "\n\n---\n\n".join(chunks)

    try:
        result = _call_llm_for_analysis(topic, context)
        logger.info("underwriting_analysis: topic=%r risk=%s", topic, result.get("risk_level"))
        return result
    except Exception as exc:
        logger.exception("underwriting_analysis: LLM call failed after retries")
        return {"error": f"Analysis LLM error: {exc}"}


# ── 3. summary_generator ──────────────────────────────────────────────────────

SUMMARY_SYSTEM_PROMPT = """\
You are a senior life insurance underwriter (Australia).
Analyse the COMPLETE application text and produce a structured summary.
Extract ONLY facts present in the text.

IMPORTANT — you MUST check for and include these specific red flags if present:
1. Compare the extended leave date with the travel section departure date — flag any difference
2. Check if applicant said No to any recreational activity but disclosed it professionally elsewhere
3. Look for any dates with invalid months (like month 0) — flag as typo
4. Flag any medical investigations advised but not yet booked
5. Flag if GP addresses differ between different sections
6. Flag missing details for scuba diving depth >40m
7. Flag if family history cancer age/relationship not stated

For disclosure_cross_check confirm status of:
- Bankruptcy, HIV, COVID, mental health, beneficiary details, payment method

Return ONLY valid JSON — no markdown, no explanation:
{
  "applicant_header": {"name": "", "dob": "", "state": "", "height_cm": 0, "weight_kg": 0, "bmi": 0.0},
  "sections": {
    "Applicant details": "",
    "Applied for cover": "",
    "Existing cover": "",
    "Modified Terms": "",
    "Claims history": "",
    "Residency": "",
    "Occupation": "",
    "Income": "",
    "Travel": "",
    "Recreation": "",
    "Alcohol": "",
    "Drug Use": "",
    "Smoking": "",
    "BMI": "",
    "Medical History": "",
    "Family History": "",
    "GP Details": ""
  },
  "red_flags": [],
  "disclosure_cross_check": []
}

"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_llm_for_summary(full_text: str) -> dict[str, Any]:
    from underwriter_agent.llm import get_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm()
    messages = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(
            content=f"APPLICATION TEXT:\n\n{full_text}\n\nReturn JSON only."
        ),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# Module-level variable to hold full PDF text for the summary tool
_full_pdf_text: str = ""


def set_full_pdf_text(text: str) -> None:
    global _full_pdf_text
    _full_pdf_text = text


@tool
def summary_generator() -> dict:
    """
    Generate the complete structured underwriting summary for the loaded application.
    Covers all 17 standard sections, red flags, and a disclosure cross-check.
    Use this when the underwriter asks for the final summary report.
    """
    if not _full_pdf_text:
        return {"error": "PDF text not loaded. Call set_full_pdf_text() first."}

    try:
        result = _call_llm_for_summary(_full_pdf_text)
        _current_turn_chunks.append(_full_pdf_text[:3000])
        logger.info("summary_generator: produced summary with %d sections",
                    len(result.get("sections", {})))
        return result
    except Exception as exc:
        logger.exception("summary_generator: LLM call failed after retries")
        return {"error": f"Summary LLM error: {exc}"}
