import argparse
import csv
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Run PLLM test_executor over many snippets and export CSV.")
    parser.add_argument(
        "-g",
        "--gists_root",
        type=str,
        required=True,
        help="Path to hard-gists root folder (contains <id>/snippet.py).",
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type=str,
        default="./eval_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument("-m", "--model", type=str, default="gemma2", help="Model name for test_executor.")
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        default="http://localhost:11434",
        help="Ollama base URL.",
    )
    parser.add_argument("-l", "--loop", type=int, default=5, help="Loop count for test_executor.")
    parser.add_argument("-r", "--range", dest="search_range", type=int, default=0, help="Python search range.")
    parser.add_argument("--rag", type=str, default="true", help="Use RAG true/false.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max snippets to run (0 = all).")
    parser.add_argument(
        "--timeout_sec",
        type=int,
        default=1800,
        help="Timeout per snippet execution in seconds.",
    )
    parser.add_argument(
        "--python_bin",
        type=str,
        default=sys.executable,
        help="Python interpreter to run test_executor.py",
    )
    return parser.parse_args()


def list_snippets(gists_root: str) -> List[str]:
    snippets = []
    for root, _, files in os.walk(gists_root):
        for f in files:
            if f == "snippet.py":
                snippets.append(os.path.join(root, f))
    snippets.sort()
    return snippets


def parse_output_file(path: str) -> Dict[str, str]:
    """
    Parse output_data_<py>.yml with a resilient regex approach.
    Returns python_version, last_error_type, total_time.
    """
    out = {"python_version": "", "last_error_type": "", "total_time": ""}
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    py_match = re.search(r"^\s*python_version:\s*(.+)$", text, flags=re.MULTILINE)
    if py_match:
        out["python_version"] = py_match.group(1).strip()

    # Support both "- error_type: X" and "error_type: X"
    all_errors = re.findall(r"^\s*(?:-\s*)?error_type:\s*(.+)$", text, flags=re.MULTILINE)
    if all_errors:
        out["last_error_type"] = all_errors[-1].strip()

    tt_match = re.search(r"^\s*total_time:\s*(.+)$", text, flags=re.MULTILINE)
    if tt_match:
        out["total_time"] = tt_match.group(1).strip()

    return out


def snapshot_output_mtimes(snippet_dir: str) -> Dict[str, float]:
    mtimes: Dict[str, float] = {}
    for name in os.listdir(snippet_dir):
        if name.startswith("output_data_") and name.endswith(".yml"):
            full = os.path.join(snippet_dir, name)
            mtimes[full] = os.path.getmtime(full)
    return mtimes


def clear_output_files(snippet_dir: str) -> None:
    for name in os.listdir(snippet_dir):
        if name.startswith("output_data_") and name.endswith(".yml"):
            try:
                os.remove(os.path.join(snippet_dir, name))
            except OSError:
                pass


def get_latest_output_file(snippet_dir: str, started_at: float, before_mtimes: Dict[str, float]) -> Optional[str]:
    candidates: List[Tuple[float, str]] = []
    for name in os.listdir(snippet_dir):
        if name.startswith("output_data_") and name.endswith(".yml"):
            full = os.path.join(snippet_dir, name)
            mtime = os.path.getmtime(full)
            previous = before_mtimes.get(full, -1.0)
            # Prefer files created/modified by this run
            if mtime > previous + 1e-6 or mtime >= started_at - 1.0:
                candidates.append((mtime, full))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def is_success(last_error_type: str) -> bool:
    # In current pipeline, "None" means no error detected.
    return last_error_type == "None"


def main():
    args = parse_args()
    gists_root = os.path.abspath(args.gists_root)
    snippets = list_snippets(gists_root)
    if args.limit > 0:
        snippets = snippets[: args.limit]

    if not snippets:
        print(f"No snippet.py files found under: {gists_root}")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.abspath(args.output_csv)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True) if os.path.dirname(output_csv) else None

    rows = []
    for idx, snippet in enumerate(snippets, start=1):
        snippet_dir = os.path.dirname(snippet)
        snippet_id = os.path.basename(snippet_dir)
        print(f"[{idx}/{len(snippets)}] Running {snippet_id}")

        # Avoid stale YAML files from previous runs affecting this row
        clear_output_files(snippet_dir)
        started_at = time.time()
        before_mtimes = snapshot_output_mtimes(snippet_dir)
        cmd = [
            args.python_bin,
            os.path.join(script_dir, "test_executor.py"),
            "-f",
            snippet,
            "-m",
            args.model,
            "-b",
            args.base,
            "-l",
            str(args.loop),
            "-r",
            str(args.search_range),
            "-ra",
            str(args.rag).lower(),
        ]

        return_code = -1
        timed_out = False
        stderr_tail = ""
        try:
            proc = subprocess.run(
                cmd,
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=args.timeout_sec,
            )
            return_code = proc.returncode
            stderr_tail = (proc.stderr or "")[-500:].replace("\n", "\\n")
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = 124
            stderr_tail = "TimeoutExpired"

        elapsed = round(time.time() - started_at, 3)
        output_file = get_latest_output_file(snippet_dir, started_at, before_mtimes)
        parsed = parse_output_file(output_file) if output_file else {
            "python_version": "",
            "last_error_type": "",
            "total_time": "",
        }

        derived_error = parsed["last_error_type"]
        if timed_out:
            derived_error = "Timeout"
        elif return_code != 0 and not derived_error:
            derived_error = "WorkerCrash"
        elif not derived_error:
            derived_error = "NoOutput"

        rows.append(
            {
                "snippet_id": snippet_id,
                "snippet_file": snippet,
                "output_file": output_file or "",
                "python_version": parsed["python_version"],
                "last_error_type": derived_error,
                "total_time": parsed["total_time"],
                "success": (return_code == 0) and is_success(parsed["last_error_type"]),
                "return_code": return_code,
                "elapsed_sec": elapsed,
                "stderr_tail": stderr_tail,
            }
        )

    fieldnames = [
        "snippet_id",
        "snippet_file",
        "output_file",
        "python_version",
        "last_error_type",
        "total_time",
        "success",
        "return_code",
        "elapsed_sec",
        "stderr_tail",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    success_count = sum(1 for r in rows if r["success"])
    print(f"\nWrote CSV: {output_csv}")
    print(f"Success: {success_count}/{total} ({(100.0 * success_count / max(total, 1)):.2f}%)")


if __name__ == "__main__":
    main()
