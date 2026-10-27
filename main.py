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
import hashlib
import logging
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.markdown import Markdown

from swarm import Swarm
from swarm.types import AnalysisReport, Response
from swarm.util import extract_json
from agents import extractor_agent, analyzer_agent, reviewer_agent


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
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent workers for batch processing (default: 5)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for OpenAI API calls (default: 3)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".swarm_cache",
        help="Directory to store cached analysis results (default: .swarm_cache)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching and force re-analysis"
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


def _save_cached_response(cache_file, response):
    if not cache_file or not response or not response.messages:
        return

    temporary_path = None
    try:
        cached_data = {
            "messages": response.messages,
            "context_variables": response.context_variables,
            "agent_name": response.agent.name if response.agent else None,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=Path(cache_file).parent,
            suffix=".tmp", delete=False,
        ) as cache_handle:
            temporary_path = Path(cache_handle.name)
            json.dump(cached_data, cache_handle, ensure_ascii=False)
        os.replace(temporary_path, cache_file)
    except (OSError, TypeError, ValueError):
        logging.getLogger(__name__).warning("Could not save analysis cache", exc_info=True)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).warning("Could not remove temporary cache file", exc_info=True)


def run_deep_analysis(massive_text, debug=False, max_turns=10, stream=False,
                      model_override=None, max_retries=3, quiet=False,
                      cache_dir=".swarm_cache", no_cache=False):
    """Execute the multi-agent analysis pipeline."""
    pipeline_agents = (extractor_agent, analyzer_agent, reviewer_agent)
    cache_file = None
    
    if cache_dir and not no_cache and not any(
        callable(agent.instructions) for agent in pipeline_agents
    ):
        cache_settings = {
            "version": 2,
            "text": massive_text,
            "max_turns": max_turns,
            "agents": [
                {
                    "name": agent.name,
                    "model": model_override or agent.model,
                    "instructions": agent.instructions,
                    "tools": [function.__name__ for function in agent.functions],
                }
                for agent in pipeline_agents
            ],
        }
        content_hash = hashlib.sha256(
            json.dumps(cache_settings, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        try:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{content_hash}.json")
        except OSError:
            logging.getLogger(__name__).warning("Analysis cache is unavailable", exc_info=True)

        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as cache_handle:
                    cached_data = json.load(cache_handle)
                cached_response = Response(
                    messages=cached_data["messages"],
                    context_variables=cached_data["context_variables"],
                    agent=next(
                        (agent for agent in pipeline_agents
                         if agent.name == cached_data["agent_name"]), None,
                    ),
                )
                if cached_response.messages and all(
                    isinstance(message, dict)
                    and message.get("role") in {"assistant", "tool", "user", "system", "developer", "function"}
                    and (message.get("content") is None or isinstance(message["content"], str))
                    for message in cached_response.messages
                ) and isinstance(cached_response.messages[-1].get("content"), str):
                    if not quiet:
                        Console().print(
                            f"[bold green][Cache Hit][/bold green] Loading analysis from {cache_file}"
                        )
                    return cached_response
            except (OSError, ValueError, KeyError, TypeError):
                pass

    if not quiet:
        console = Console()
        console.print("[bold blue][System][/bold blue] Starting Swarm-based multi-agent deep analysis...")
        console.print("[bold red][Warning][/bold red] This process involves stateless context handoff "
                      "and will consume significant tokens.")

    client = Swarm(max_retries=max_retries)

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
        
        _save_cached_response(cache_file, final_response)
                
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
        
        _save_cached_response(cache_file, response)
                
        return response


def run_batch(args):
    """Run analysis on multiple files in batch mode."""
    console = Console()
    files = find_batch_files(args.batch, args.pattern, args.max_size)
    if not files:
        console.print(f"[yellow]No files matching '{args.pattern}' found in {args.batch}[/yellow]")
        return

    console.print(f"[bold cyan][Batch][/bold cyan] Found {len(files)} files to analyze")
    console.print(f"[bold cyan][Batch][/bold cyan] Pattern: {args.pattern}")
    console.print(f"[bold cyan][Batch][/bold cyan] Max file size: {args.max_size} bytes")
    console.print(f"[bold cyan][Batch][/bold cyan] Workers: {args.workers}")
    console.print()

    results = []

    def process_file(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError) as e:
            return {"file": filepath, "error": str(e)}

        if not content.strip():
            return {"file": filepath, "error": "empty file"}

        start_time = time.time()
        response = run_deep_analysis(
            content,
            debug=args.debug,
            max_turns=args.max_turns,
            stream=False,
            model_override=args.model,
            max_retries=args.max_retries,
            quiet=True,
            cache_dir=args.cache_dir,
            no_cache=args.no_cache,
        )
        elapsed = time.time() - start_time

        result_entry = {
            "file": filepath,
            "chars": len(content),
            "elapsed_seconds": round(elapsed, 2),
            "raw_output": "",
        }

        if response and hasattr(response, "messages"):
            final_content = response.messages[-1].get("content", "")
            result_entry["raw_output"] = final_content

            parsed = extract_json(final_content)
            if parsed:
                try:
                    report = AnalysisReport(**parsed)
                    result_entry["total_issues"] = report.total_issues
                    result_entry["critical"] = report.critical_count
                    result_entry["high"] = report.high_count
                except Exception:
                    pass
        else:
            result_entry["error"] = "no response"

        if args.output and "error" not in result_entry:
            os.makedirs(args.output, exist_ok=True)
            basename = Path(filepath).stem + "_report.json"
            report_path = os.path.join(args.output, basename)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(result_entry, f, indent=2, ensure_ascii=False)

        return result_entry

    with Progress() as progress:
        task = progress.add_task("[cyan]Analyzing files...", total=len(files))
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {executor.submit(process_file, f): f for f in files}
            for future in as_completed(future_to_file):
                res = future.result()
                results.append(res)
                progress.advance(task)

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
        console.print(f"\n[bold green][Batch][/bold green] Summary saved to: {summary_path}")

    console.print(f"\n[bold green][Batch][/bold green] Complete: {summary['analyzed']}/{summary['total_files']} files analyzed")
    if summary["errors"] > 0:
        console.print(f"[bold red][Batch] Errors: {summary['errors']}[/bold red]")

    # Print Table
    table = Table(title="Batch Analysis Summary")
    table.add_column("File", justify="left", style="cyan", no_wrap=True)
    table.add_column("Time (s)", justify="right", style="magenta")
    table.add_column("Issues", justify="right", style="red")
    table.add_column("Status", justify="center", style="green")

    for r in results:
        fname = os.path.basename(r["file"])
        if "error" in r:
            table.add_row(fname, "-", "-", f"[red]Error: {r['error']}[/red]")
        else:
            issues = str(r.get("total_issues", "?"))
            if r.get("critical", 0) > 0:
                issues += f" [bold red]({r['critical']}C)[/bold red]"
            table.add_row(fname, str(r["elapsed_seconds"]), issues, "[green]OK[/green]")

    console.print(table)


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
        max_retries=args.max_retries,
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
    )

    if response and hasattr(response, "messages"):
        final_content = response.messages[-1].get("content", "")
        console = Console()
        console.print("\n[bold blue][System][/bold blue] Analysis complete. Final report:")
        console.print("=" * 60)
        console.print(Markdown(final_content))
        console.print("=" * 60)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(final_content)
            print(f"[System] Report saved to: {args.output}")
    else:
        print("[System] Analysis complete (streaming mode).")


if __name__ == "__main__":
    main()
