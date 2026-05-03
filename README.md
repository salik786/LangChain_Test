What I built
A command-line underwriting assistant that lets an underwriter have a real conversation with an insurance application PDF. You ask questions in plain English. It searches the document, applies underwriting rules, and generates a structured report — all without making things up.

The five-turn demo below shows it working end to end against the provided Input_Sample.pdf (John Smith, policy #168460-43865).


Running it
git clone https://github.com/salik786/nobleoak-underwriting-agent
cd nobleoak-underwriting-agent
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
python run.py --record-id 168460-43865 --pdf Input_Sample.pdf

Python 3.11+, one API key, no other setup required.


How it works
There are six files. Each has one job.

run.py	CLI entry point. Loads the PDF, boots the graph, runs the chat loop.
ingestion.py	Reads the PDF with pypdf, splits into 24 chunks (~500 tokens each) (just for demo), embeds with OpenAI, stores in a FAISS index. Runs once on startup.
llm.py	Single place to get the LLM client. Auto-detects OpenAI, Anthropic or Azure from environment variables.
tools.py	Three @tool functions the agent can call: document_search (FAISS retrieval), underwriting_analysis (retrieval + domain-rules LLM call), summary_generator (full 17-section report from complete PDF text).
graph.py	The main StateGraph. Agent calls tools in a loop, then hands the draft response to the judge subgraph. A conditional edge either emits the response, asks the agent to revise once, or emits with a visible warning.
judges.py	A separate compiled subgraph. Three judge nodes run in parallel via LangGraph native fan-out: factuality, completeness, consistency. Results merge via operator.add reducer into a single verdict.


Per-turn flow
User question
  → agent_node  (LLM decides which tool to call)
  → tools_node  (FAISS search + optional LLM analysis)
  → agent_node  (loop until no more tool calls)
  → judge subgraph  [factuality | completeness | consistency] (parallel)
  → merge_verdicts
  → PASS  → emit to user
  → FAIL + first attempt  → revise_node → agent_node
  → FAIL + already revised  → emit with ■ Judge flagged warning


State design
AgentState is a TypedDict with explicit reducers on every field that can be written by multiple nodes at once.

messages:       Annotated[list[BaseMessage], add_messages]
source_chunks:  Annotated[list[str], operator.add]
draft_response: str | None
judge_verdict:  dict | None
revision_count: int

source_chunks uses operator.add so tool calls within a turn append rather than overwrite — which matters especially during the parallel judge fan-out, where three nodes write to the same field at the same time.


Why the judges run in parallel
All three judge nodes share a common predecessor (fan_out) and have no data dependency between them. LangGraph places them in the same superstep and executes them concurrently. That cuts validation latency from ~6 seconds (sequential) to ~2 seconds.

The operator.add reducer on judge_results means partial writes from each judge accumulate rather than overwrite — the merge node sees all three results once they complete.


Error handling
●Retry with exponential backoff (tenacity) on all LLM calls — up to 3 attempts, 2–10s wait
●Graceful 'not in the application' response when FAISS returns no chunks above threshold
●Tool exceptions surfaced as assistant messages, not process crashes
●JSON-lines structured logging to stderr on every node transition and tool invocation



The five-turn conversation
Each turn exercises a different part of the system.




Red flags found in Input_Sample.pdf
● Travel date conflict — extended leave states 17/04/2026, travel section states 17/06/2026
● Pending endoscopy for GORD/reflux — advised but not yet booked (deferral trigger)
● PSA test date recorded as 18/0/2024 — invalid month, requires correction
● Professional bicycle racing disclosed but applicant answered No to recreational cycling checklist
● GP addresses differ between sections — 20/25 Smith Road vs 200 Smith Street
● Scuba diving >40m — frequency, locations and decompression practices not disclosed
● Family history breast cancer — relationship and age at diagnosis not stated
● Smoker loading required — less than 10 cigarettes per day


What I'd change for production
Checkpointer	MemorySaver (in-process dict) → PostgresSaver. Conversations survive restarts, support multi-worker deployments, and are auditable.
Vector store	FAISS in-memory → pgvector (via langchain-postgres) or Pinecone. Persistent, filterable, scales to millions of chunks without rebuilding on every start.
Judge models	Different judges can use advance models. Same reasoning quality for classification tasks, ~10x cheaper per turn.
Serving	CLI loop → FastAPI /chat endpoint with .astream_events for token streaming.
Observability	stderr JSON-lines → OpenTelemetry to Datadog. Structured traces, latency dashboards, cost monitoring per record.
PDF parsing	pypdf (text only) → pdfplumber for tables, pytesseract for scanned pages. Checkbox values are inferred from text currently — not always reliable.


Tech stack
Orchestration	langgraph 0.2.x — StateGraph, MemorySaver, ToolNode
LLM	langchain-openai — GPT-4o, temperature=0
Embeddings	text-embedding-3-small (OpenAI)
Vector store	FAISS in-memory (langchain-community)
PDF parsing	pypdf 4.x
Retry/backoff	tenacity
Tracing	LangSmith (optional)
Python	3.11+


A note on the approach
I kept the code straightforward on purpose. The parallel fan-out is worth more than the revision loop, so I made sure that part is clean and easy to follow rather than hiding it behind abstractions. The three edges from fan_out in judges.py are the core of the submission — everything else supports them.

Happy to walk through any part of it in detail.
