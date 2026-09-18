"""
Swarm-Deep-Analyzer: Stateless Multi-Agent Reasoning Engine

Entry point for the deep analysis pipeline.
Orchestrates Extractor -> Analyzer -> Reviewer agent chain
via Swarm's stateless context handoff mechanism.

Usage:
    python main.py                          # Run with built-in sample text
    python main.py --file input.txt         # Analyze a file
    python main.py --text "your text here"  # Analyze inline text
    python main.py --demo                   # Run interactive demo loop
    python main.py --batch ./src --pattern "*.py"  # Batch analyze files
"""

import argparse
import glob
import os
import sys
import json
import time
from pathlib import Path

from swarm import Swarm
from swarm.types import AnalysisReport
from swarm.util import extract_json
from agents import extractor_agent


BUILT_IN_SAMPLE = """
Algorithm Analysis Report: Custom Segment Tree Implementation

This codebase implements a persistent segment tree for range minimum queries
over a dynamically-updated integer array. The core data structure uses a
node-pool allocation strategy with an implicit free-list.

Key architectural observations:
1. Memory Management: Each update creates O(log n) new nodes via malloc().
   No garbage collection or reference counting exists for old nodes.
   Over thousands of updates, this will exhaust available heap memory.

2. Pointer Arithmetic: The left/right child pointers are stored as integer
   offsets into the node pool array, not as raw pointers. This is intentional
   for serialization but creates a risk: if the pool is reallocated (via
   realloc), all previously-stored offsets remain valid, but any external
   code holding raw pointers to pool elements will have dangling references.

3. Algorithm Complexity: The query function claims O(log n) complexity.
   However, the recursive implementation does not use tail-call optimization.
   For trees of depth > 10,000, this risks stack overflow. An iterative
   approach would be safer and equally fast.

4. Concurrency: The structure is not thread-safe. Concurrent reads during
   an update may observe partially-constructed nodes (the children are
   written before the parent metadata). A memory barrier or double-checked
   locking pattern is needed for multi-threaded usage.

5. Edge Cases: Empty array queries return INT_MIN as sentinel. This is
   not documented and could confuse callers who legitimately store INT_MIN
   values in the array.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Swarm-Deep-Analyzer: Multi-agent deep reasoning pipeline"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a text file to analyze"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Inline text to analyze"
    )
    parser.add_argument(
        "--batch", "-b",
        type=str,
        help="Directory path for batch processing (analyzes all matching files)"
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default="*.py",
        help="File pattern for batch mode (default: *.py)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=100000,
        help="Max file size in bytes for batch mode (default: 100KB)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Launch interactive demo loop"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to observe token handoff logs"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum number of agent turns (default: 10)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Write final report to this file (single file mode) or directory (batch mode)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming output"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model for all agents (e.g., gpt-4o-mini)"
    )
    return parser.parse_args()


def load_input(args):
    """Resolve input text from args, file, or built-in sample."""
    if args.text:
        return args.text
    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    return BUILT_IN_SAMPLE


def find_batch_files(directory, pattern, max_size):
    """Find files matching pattern in directory, respecting size limits."""
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}", file=sys.stderr)
        sys.exit(1)

    search_pattern = os.path.join(directory, "**", pattern)
    files = glob.glob(search_pattern, recursive=True)

    filtered = []
    for f in sorted(files):
        if os.path.isfile(f):
            size = os.path.getsize(f)
            if size <= max_size and size > 0:
                filtered.append(f)

    return filtered


def run_deep_analysis(massive_text, debug=False, max_turns=10, stream=False,
                      model_override=None):
    """Execute the multi-agent analysis pipeline."""
    print("[System] Starting Swarm-based multi-agent deep analysis...")
    print("[Warning] This process involves stateless context handoff "
          "and will consume significant tokens.")

    client = Swarm()

    if stream:
        response = client.run(
            agent=extractor_agent,
            messages=[{
                "role": "user",
                "content": (
                    "Please perform a deep structured analysis "
                    "of the following material:\n\n"
                    f"{massive_text}"
                )
            }],
            debug=debug,
            max_turns=max_turns,
            stream=True,
            model_override=model_override,
        )
        final_response = None
        for chunk in response:
            if "content" in chunk and chunk["content"]:
                print(chunk["content"], end="", flush=True)
            if "response" in chunk:
                final_response = chunk["response"]
        print()
        return final_response
    else:
        response = client.run(
            agent=extractor_agent,
            messages=[{
                "role": "user",
                "content": (
                    "Please perform a deep structured analysis "
                    "of the following material:\n\n"
                    f"{massive_text}"
                )
            }],
            debug=debug,
            max_turns=max_turns,
            model_override=model_override,
        )
        return response


def run_batch(args):
    """Run analysis on multiple files in batch mode."""
    files = find_batch_files(args.batch, args.pattern, args.max_size)
    if not files:
        print(f"No files matching '{args.pattern}' found in {args.batch}")
        return

    print(f"[Batch] Found {len(files)} files to analyze")
    print(f"[Batch] Pattern: {args.pattern}")
    print(f"[Batch] Max file size: {args.max_size} bytes")
    print()

    results = []
    for i, filepath in enumerate(files, 1):
        print(f"[Batch] ({i}/{len(files)}) Analyzing: {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError) as e:
            print(f"[Batch] Skipping {filepath}: {e}")
            results.append({"file": filepath, "error": str(e)})
            continue

        if not content.strip():
            print(f"[Batch] Skipping {filepath}: empty file")
            results.append({"file": filepath, "error": "empty file"})
            continue

        start_time = time.time()
        response = run_deep_analysis(
            content,
            debug=args.debug,
            max_turns=args.max_turns,
            stream=False,
            model_override=args.model,
        )
        elapsed = time.time() - start_time

        if response and hasattr(response, "messages"):
            final_content = response.messages[-1].get("content", "")

            # Try to parse structured report
            parsed = extract_json(final_content)
            report = None
            if parsed:
                try:
                    report = AnalysisReport(**parsed)
                except Exception:
                    pass

            result_entry = {
                "file": filepath,
                "chars": len(content),
                "elapsed_seconds": round(elapsed, 2),
                "raw_output": final_content,
            }
            if report:
                result_entry["total_issues"] = report.total_issues
                result_entry["critical"] = report.critical_count
                result_entry["high"] = report.high_count
                print(f"[Batch]   -> {report.total_issues} issues "
                      f"({report.critical_count} critical, {report.high_count} high) "
                      f"in {elapsed:.1f}s")
            else:
                print(f"[Batch]   -> Analysis complete in {elapsed:.1f}s")

            results.append(result_entry)

            # Save individual report if output dir specified
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                basename = Path(filepath).stem + "_report.json"
                report_path = os.path.join(args.output, basename)
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(result_entry, f, indent=2, ensure_ascii=False)
        else:
            results.append({"file": filepath, "error": "no response"})

    # Save summary
    summary = {
        "total_files": len(files),
        "analyzed": len([r for r in results if "error" not in r]),
        "errors": len([r for r in results if "error" in r]),
        "results": results,
    }

    if args.output:
        summary_path = os.path.join(args.output, "batch_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[Batch] Summary saved to: {summary_path}")

    print(f"\n[Batch] Complete: {summary['analyzed']}/{summary['total_files']} files analyzed")
    if summary["errors"] > 0:
        print(f"[Batch] Errors: {summary['errors']}")


def run_demo(args):
    """Launch the interactive REPL demo."""
    from swarm.repl import run_demo_loop
    print("Launching interactive demo with Extractor agent...")
    run_demo_loop(
        starting_agent=extractor_agent,
        debug=args.debug,
        stream=args.stream,
    )


def main():
    args = parse_args()

    if args.demo:
        run_demo(args)
        return

    if args.batch:
        run_batch(args)
        return

    text = load_input(args)

    print(f"[System] Input length: {len(text)} characters")
    print(f"[System] Estimated tokens: ~{len(text) // 4}")
    print()

    response = run_deep_analysis(
        text,
        debug=args.debug,
        max_turns=args.max_turns,
        stream=args.stream,
        model_override=args.model,
    )

    if response and hasattr(response, "messages"):
        final_content = response.messages[-1].get("content", "")
        print("\n[System] Analysis complete. Final report:")
        print("=" * 60)
        print(final_content)
        print("=" * 60)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(final_content)
            print(f"[System] Report saved to: {args.output}")
    else:
        print("[System] Analysis complete (streaming mode).")


if __name__ == "__main__":
    main()
