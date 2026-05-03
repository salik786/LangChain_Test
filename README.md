# NobleOak Underwriting Agent

> A conversational AI assistant for life insurance underwriting. Point it at an application PDF, ask questions in plain English, and get answers grounded in the document no hallucination, no guessing.

Built with LangGraph for Exercise 3 of the NobleOak AI assessment.

---

## Quick Start

```bash
git clone https://github.com/salik786/nobleoak-underwriting-agent
cd nobleoak-underwriting-agent
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
python run.py --record-id 168460-43865 --pdf Input_Sample.pdf
```

Python 3.11+, one API key, no other setup required.

---

## What I Built

A command-line underwriting assistant that lets an underwriter have a real conversation with an insurance application PDF. You ask questions in plain English. It searches the document, applies underwriting rules, and generates a structured report all without making things up.

The five-turn demo below shows it working end to end against the provided `Input_Sample.pdf` (John Smith, policy #168460-43865).

---

## The Five-Turn Conversation

Each turn exercises a different part of the system.

| Turn | Question | Tool Called | What It Tests |
|------|----------|-------------|---------------|
| 1 | Summarise this application | `document_search` | Basic retrieval + multi-section synthesis |
| 2 | What are the medical red flags? | `underwriting_analysis` | Domain rules — cancer, cholesterol, alcohol |
| 3 | Tell me more about the melanoma history | `document_search` | Multi-turn context — builds on Turn 2 |
| 4 | Are there any date inconsistencies? | `underwriting_analysis` | Cross-section inconsistency detection |
| 5 | Generate the final underwriting summary | `summary_generator` | Full 17-section structured report |



---

## How It Works

### File Structure

There are six files. Each has one job.

| File | What it does |
|------|-------------|
| `run.py` | CLI entry point. Loads the PDF, boots the graph, runs the chat loop. |
| `ingestion.py` | Reads the PDF with pypdf, splits into chunks (~500 tokens each), embeds with OpenAI, stores in a FAISS index. Runs once on startup. |
| `llm.py` | Single place to get the LLM client. Auto-detects OpenAI, Anthropic or Azure from environment variables. |
| `tools.py` | Three `@tool` functions the agent can call: `document_search` (FAISS retrieval), `underwriting_analysis` (retrieval + domain-rules LLM call), `summary_generator` (full 17-section report). |
| `graph.py` | The main StateGraph. Agent calls tools in a loop, then hands the draft response to the judge subgraph. A conditional edge either emits the response, asks the agent to revise once, or emits with a visible warning. |
| `judges.py` | A separate compiled subgraph. Three judge nodes run in parallel via LangGraph native fan-out: factuality, completeness, consistency. Results merge via `operator.add` reducer into a single verdict. |

### Per-Turn Flow

```
User question
  → agent_node         (LLM decides which tool to call)
  → tools_node         (FAISS search + optional LLM analysis)
  → agent_node         (loop until no more tool calls)
  → judge subgraph     [factuality | completeness | consistency]  ← parallel
  → merge_verdicts
  → PASS               → emit to user
  → FAIL (1st time)    → revise_node → agent_node
  → FAIL (revised)     → emit with ■ Judge flagged warning
```

### State Design

`AgentState` is a `TypedDict` with explicit reducers on every field that can be written by multiple nodes at once.

```python
class AgentState(TypedDict):
    messages:       Annotated[list[BaseMessage], add_messages]
    record_id:      str
    draft_response: str | None
    source_chunks:  Annotated[list[str], operator.add]
    judge_verdict:  dict | None
    revision_count: int
```

`source_chunks` uses `operator.add` so tool calls within a turn append rather than overwrite — which matters especially during the parallel judge fan-out, where three nodes write to the same field at the same time.

### Why the Judges Run in Parallel

All three judge nodes share a common predecessor (`fan_out`) and have no data dependency between them. LangGraph places them in the same superstep and executes them concurrently. That cuts validation latency from ~6 seconds (sequential) to ~2 seconds.

The `operator.add` reducer on `judge_results` means partial writes from each judge accumulate rather than overwrite — the merge node sees all three results once they complete.

```
fan_out → factuality_judge   ─┐
        → completeness_judge  ─┼─► merge_verdicts → END
        → consistency_judge   ─┘
```

---

## Red Flags Found in Input_Sample.pdf

The agent identified these from the John Smith application (#168460-43865):

- **Travel date conflict** — extended leave states 17/04/2026, travel section states 17/06/2026
- **Pending endoscopy** — advised for GORD/reflux but not yet booked (standard deferral trigger)
- **PSA date typo** — recorded as `18/0/2024`, invalid month, likely 18/02/2024
- **Cycling inconsistency** — answered No to recreational cycling but disclosed professional bicycle racing
- **GP address discrepancy** — 20/25 Smith Road (melanoma GP) vs 200 Smith Street (regular clinic)
- **Scuba diving >40m** — frequency, locations and decompression practices not disclosed
- **Family history incomplete** — breast cancer in first-degree relative, age and relationship not stated
- **Smoker loading required** — less than 10 cigarettes per day

---

## Error Handling

- **Retry with exponential backoff** (`tenacity`) on all LLM calls — up to 3 attempts, 2–10s wait
- **Graceful fallback** — `document_search` returns "not in the application" when FAISS finds nothing relevant, rather than guessing
- **Tool exception surfacing** — exceptions from tools appear as assistant messages, not process crashes
- **Structured logging** — JSON-lines to stderr on every node transition and tool invocation

---

## What I'd Change for Production

| Component | This Assessment | Production |
|-----------|----------------|------------|
| Checkpointer | `MemorySaver` (in-process) | `PostgresSaver` — survives restarts, multi-worker safe, auditable |
| Vector store | FAISS in-memory | `pgvector` or Pinecone — persistent, filterable, no rebuild on restart |
| Judge models | GPT-4o for all judges | Lighter models for judges — same quality for classification, cheaper |
| Serving | CLI loop | FastAPI `/chat` with `.astream_events` for token streaming |
| Observability | stderr JSON-lines | OpenTelemetry → Datadog — traces, latency dashboards, cost per record |
| PDF parsing | `pypdf` (text only) | `pdfplumber` for tables, `pytesseract` for scanned pages |

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Orchestration | `langgraph` 0.2.x — StateGraph, MemorySaver, ToolNode |
| LLM | `langchain-openai` — GPT-4o, temperature=0 |
| Embeddings | `text-embedding-3-small` (OpenAI) |
| Vector store | FAISS in-memory (`langchain-community`) |
| PDF parsing | `pypdf` 4.x |
| Retry / backoff | `tenacity` |
| Tracing | LangSmith (optional) |
| Python | 3.11+ |

---



## A Note on the Approach

I kept the code straightforward on purpose. The parallel fan-out is worth more than the revision loop, so I made sure that part is clean and easy to follow rather than hiding it behind abstractions. The three edges from `fan_out` in `judges.py` are the core of the submission — everything else supports them.

Happy to walk through any part of it in detail.
