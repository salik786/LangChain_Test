#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}',
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("run")


def _load_env() -> None:
    """Load .env file if present (silently skip if python-dotenv not installed)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info('"Loading .env"')
    except ImportError:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NobleOak Underwriting Agent — multi-turn conversational assistant"
    )
    parser.add_argument(
        "--record-id",
        required=True,
        help="Application record ID (used as the conversation thread ID)",
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the insurance application PDF",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Generate and print the underwriting summary then exit",
    )
    parser.add_argument(
        "--save-summary",
        metavar="FILE",
        help="Path to save the generated summary as JSON (e.g. summary.json)",
    )
    return parser.parse_args()


def _print_banner(record_id: str, pdf_path: str) -> None:
    print("\n" + "═" * 65)
    print("  NobleOak Underwriting Agent")
    print(f"  Record: {record_id}  |  PDF: {pdf_path}")
    print("═" * 65)
    print("  Commands: type your question, or 'quit' / 'exit' to stop.")
    print("  Tip: try asking for a summary, medical red flags, or date checks.\n")


def _extract_assistant_text(event: dict) -> str | None:
    """Pull the final assistant text out of a stream event."""
    for node_output in event.values():
        if not isinstance(node_output, dict):
            continue
        messages = node_output.get("messages", [])
        for msg in reversed(messages):
            # Only emit messages that come from the 'emit' node (final reply)
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                return msg.content
    return None


def main() -> None:
    _load_env()
    args = _parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    print(f"\n[*] Loading PDF: {pdf_path} …", end=" ", flush=True)
    from underwriter_agent.ingestion import build_retriever, _load_pdf_text
    retriever = build_retriever(pdf_path)
    full_text = _load_pdf_text(pdf_path)
    print("done.")

    print("[*] Building graph …", end=" ", flush=True)
    from underwriter_agent.graph import build_graph
    graph = build_graph(retriever=retriever, full_pdf_text=full_text)
    print("done.\n")

    config = {"configurable": {"thread_id": args.record_id}}

    # ── Summary-only mode ─────────────────────────────────────────────────────
    if args.summary_only or args.save_summary:
        print("[*] Generating underwriting summary …\n")
        summary_prompt = "Generate the final underwriting summary."
        result_text = None

        for event in graph.stream(
            {
                "messages": [("user", summary_prompt)],
                "record_id": args.record_id,
                "draft_response": None,
                "source_chunks": [],
                "judge_verdict": None,
                "revision_count": 0,
            },
            config,
            stream_mode="updates",
        ):
            text = _extract_assistant_text(event)
            if text:
                result_text = text

        if result_text:
            print(result_text)
            # Try to extract the JSON dict from the tool call result
            if args.save_summary:
                # The summary is embedded in the AI response; we re-call the tool
                # directly to get the raw dict for JSON serialisation
                from underwriter_agent.tools import _call_llm_for_summary
                try:
                    summary_dict = _call_llm_for_summary(full_text)
                    out_path = Path(args.save_summary)
                    out_path.write_text(json.dumps(summary_dict, indent=2))
                    print(f"\n[*] Summary saved to: {out_path}")
                except Exception as exc:
                    logger.warning('"summary JSON save failed: %s"', exc)

        if args.summary_only:
            return

    # ── Interactive chat loop ─────────────────────────────────────────────────
    _print_banner(args.record_id, str(pdf_path))

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[*] Session ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("[*] Goodbye.")
            break

        print()  # blank line before response

        last_text: str | None = None
        try:
            for event in graph.stream(
                {
                    "messages": [("user", user_input)],
                    "record_id": args.record_id,
                    "draft_response": None,
                    "source_chunks": [],
                    "judge_verdict": None,
                    "revision_count": 0,
                },
                config,
                stream_mode="updates",
            ):
                text = _extract_assistant_text(event)
                if text:
                    last_text = text

        except Exception as exc:
            logger.exception('"graph stream error"')
            print(f"[Agent Error] {exc}\n")
            continue

        if last_text:
            print(f"Agent: {last_text}\n")
        else:
            print("[Agent produced no output for this turn]\n")


if __name__ == "__main__":
    main()
