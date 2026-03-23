import argparse
import json
import os
import subprocess
import sys
import urllib.request
from typing import Dict, Tuple


def run_cmd(cmd: list) -> Tuple[bool, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return True, out
    except Exception as e:
        return False, str(e)


def check_ollama(base_url: str) -> Tuple[bool, str]:
    try:
        url = base_url.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        data = json.loads(body)
        models = [m.get("name", "") for m in data.get("models", [])]
        return True, f"reachable ({len(models)} models): {', '.join(models[:5])}"
    except Exception as e:
        return False, str(e)


def check_path_exists(path: str) -> Tuple[bool, str]:
    if os.path.exists(path):
        return True, path
    return False, f"Missing path: {path}"


def main():
    parser = argparse.ArgumentParser(description="Quick environment parity check for PLLM.")
    parser.add_argument(
        "--gists_root",
        type=str,
        default="../../hard-gists",
        help="Expected hard-gists root folder.",
    )
    parser.add_argument(
        "--ollama_base",
        type=str,
        default="http://localhost:11434",
        help="Ollama base URL to check.",
    )
    args = parser.parse_args()

    checks: Dict[str, Tuple[bool, str]] = {}
    checks["python_version"] = run_cmd([sys.executable, "--version"])
    checks["python_path"] = (True, sys.executable)
    checks["docker_version"] = run_cmd(["docker", "--version"])
    checks["docker_ps"] = run_cmd(["docker", "ps", "--format", "{{.Names}}"])
    checks["ollama_version"] = run_cmd(["ollama", "--version"])
    checks["ollama_api"] = check_ollama(args.ollama_base)
    checks["gists_root"] = check_path_exists(os.path.abspath(args.gists_root))

    print("PLLM Environment Check")
    print("=" * 60)
    failed = 0
    for name, (ok, msg) in checks.items():
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: {msg}")
        if not ok:
            failed += 1
    print("=" * 60)
    if failed:
        print(f"Result: {failed} check(s) failed. Fix these before running eval.")
        sys.exit(1)
    print("Result: all checks passed.")


if __name__ == "__main__":
    main()
