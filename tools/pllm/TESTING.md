# How to test the PLLM pipeline

## Prerequisites

1. **Docker** – for building/running the evaluator and snippet environments.
2. **LangGraph** – the resolution flow is implemented with LangGraph. Install with: `pip install langgraph` (or use the project Pipfile: `pipenv install`).
3. **Ollama** – for the LLM (e.g. Gemma2). Install from https://ollama.com/ then:
   ```bash
   ollama pull gemma2
   ollama serve   # or leave it running in the background
   ```
3. **Hard-gists dataset** – extract the competition snippets so the container can see them:
   - Get `hard-gists.tar.gz` (from the competition / dataset source).
   - From the **repository root** (parent of `tools/`), create and extract:
     ```bash
     mkdir -p hard-gists
     tar -xzf hard-gists.tar.gz -C hard-gists
     ```
   - The `docker-compose.yml` mounts `../../hard-gists` as `/gists` inside the container, so paths like `/gists/<id>/snippet.py` will work.

---

## Option A: Full test with Docker Compose (recommended)

**Start from the repository root** (the folder that contains `tools/`, e.g. `fse-aiware-python-dependencies`). If you're elsewhere, run:
   ```bash
   cd /path/to/fse-aiware-python-dependencies
   ```

1. **Create `.env` in `tools/pllm`** (run these from the repo root):
   ```bash
   cd tools/pllm
   echo "USER=$(whoami)" > .env
   echo "UID=$(id -u)" >> .env
   echo "GID=$(id -g)" >> .env
   # Works on both macOS and Linux:
   echo "DOCKER_GID=$(stat -f '%g' /var/run/docker.sock 2>/dev/null || stat -c '%g' /var/run/docker.sock)" >> .env
   ```
   If you get "No such file or directory", you're not in the repo root—run `cd` to the repo root first, then run the block again. If Docker isn't installed or the socket is missing, create `.env` without the last line and set `DOCKER_GID=999` for now.

2. **Build and start** (Ollama + PLLM container):
   ```bash
   cd tools/pllm
   ./build.sh
   docker compose up -d
   ```

3. **Run a single snippet** (inside the `pllm-test` container):
   - If Ollama runs **on the host**, use `host.docker.internal`:
     ```bash
     docker exec -it pllm-test python test_executor.py \
       -f '/gists/0a2ac74d800a2eff9540/snippet.py' \
       -m gemma2 \
       -b 'http://host.docker.internal:11434' \
       -l 5 -r 0
     ```
   - If Ollama runs **in the same compose** (container name `ollama`):
     ```bash
     docker exec -it pllm-test python test_executor.py \
       -f '/gists/0a2ac74d800a2eff9540/snippet.py' \
       -m gemma2 \
       -b 'http://ollama:11434' \
       -l 5 -r 0
     ```

4. **Parameters**:
   - `-f` – path to `snippet.py` (use any `<id>/snippet.py` under `/gists` if you have the dataset).
   - `-m` – Ollama model name (e.g. `gemma2`, `phi3:medium`).
   - `-b` – Ollama base URL.
   - `-l` – max resolution loops (e.g. `5` or `10`).
   - `-r` – Python version search range (0 = only LLM-picked version; 1 = ±1 version).
   - `-t` – temperature (default 0.7).
   - `-ra` – RAG on/off (default true).

Success: the run finishes and writes an `output_data_<version>.yml` next to the snippet. One process will set the shared success event so other Python-version processes exit early.

---

## Option B: Test without Docker (host only)

If you only want to sanity-check **AST import extraction** and **error parsing** (no LLM, no Docker builds):

From `tools/pllm`:

```bash
# AST + error parser (no Ollama/Docker)
python -c "
from helpers.deps_scraper import DepsScraper
from helpers.ollama_helper_tester import parse_could_not_find_version_error

# Test AST extraction on a real file
d = DepsScraper(logging=True)
imports = d.extract_imports_ast('test_executor.py')
print('AST imports from test_executor.py:', imports[:15])

# Test pip error parsing
err = 'ERROR: Could not find a version that satisfies the requirement djangorestframework==3.15.2 (from versions: 0.1, 2.0.0, 3.0.0)'
parsed = parse_could_not_find_version_error(err)
print('Parsed error:', parsed)
"
```

This verifies that the new helpers run; full resolution still requires Docker + Ollama as in Option A.

---

## Option C: Build script only (no compose)

From the host:

```bash
cd tools/pllm
./build.sh
```

Then run the image with the socket and gists mounted (adjust paths as needed):

```bash
docker run -it --rm \
  -v /var/run/docker.sock:/var/run/docker.sock:rw \
  -v $(pwd):/app \
  -v /path/to/hard-gists:/gists:ro \
  --name pllm-test \
  pllm:latest \
  bash
```

Inside the container, run the same `python test_executor.py ...` command as in Option A (use `-b http://host.docker.internal:11434` if Ollama is on the host).

---

## Batch evaluation with CSV (improved pipeline)

Use `eval_runner.py` to run the improved `test_executor` (which uses `resolution_graph.py`) across many snippets and export a CSV.

From `tools/pllm`:

```bash
python eval_runner.py \
  -g "/path/to/hard-gists" \
  -o "./eval_results.csv" \
  -m gemma2 \
  -b "http://localhost:11434" \
  -l 5 -r 0 --rag true --limit 50
```

CSV columns:
- `snippet_id`, `snippet_file`, `output_file`
- `python_version`
- `last_error_type`
- `total_time` (from output YAML when present)
- `success` (true when `last_error_type == None`)
- `return_code`, `elapsed_sec`

Notes:
- `-g` must point to the root folder that contains `<id>/snippet.py`.
- `--limit 0` means run all snippets.
- This runner uses your current Python interpreter (`sys.executable`) unless you pass `--python_bin`.

---

## Troubleshooting

- **Docker permission denied**: Ensure the socket GID in `.env` matches `stat -c '%g' /var/run/docker.sock` (Linux) or `stat -f '%g' /var/run/docker.sock` (macOS). See `tools/pllm/DOCKER_SETUP.md`.
- **Ollama connection refused**: If the evaluator runs in Docker, use `http://host.docker.internal:11434` (macOS/Windows) or the Ollama container name (e.g. `http://ollama:11434`) when Ollama is in the same compose.
- **No such file: '/gists/...'**: Extract `hard-gists.tar.gz` so that `hard-gists/<id>/snippet.py` exists and the volume is mounted as `/gists`.
- **Module/import errors when running test_executor.py**: Run from `tools/pllm` (working dir `/app` in the container) so `helpers` and `test_executor` resolve.
