"""
LangGraph-based resolution flow for PLLM.
Matches workflow: infer_python → parse_imports → normalize_modules → query_py1 → select_versions
→ generate_dockerfiles → docker_build → docker_run → classify_errors → resolve_error
→ decide_strategy → (docker_build retry | finalize) → finalize → end.
"""
from typing import TypedDict, Literal, Any, Optional

from langgraph.graph import StateGraph, END

from helpers.build_dockerfile import DockerHelper


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ResolutionState(TypedDict, total=False):
    llm_eval: dict
    parsed_imports: list
    normalized_modules: list
    error_handler: dict
    docker_output: str
    output: Optional[dict]
    error_type: str
    strategy: str
    last_error_signature: str
    stagnation_count: int
    build_complete: bool
    run_complete: bool
    loop: int
    file_to_open: str
    docker_helper: Any
    project_dir: str
    dir_name: str
    project_file: str


def default_error_handler() -> dict:
    return {
        "previous": "",
        "error_modules": {},
        "ImportError": 0,
        "ModuleNotFound": 0,
        "VersionNotFound": 0,
        "DependencyConflict": 0,
        "AttributeError": 0,
        "NonZeroCode": 0,
        "SyntaxError": 0,
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_resolution_graph(
    executor,
    ollama_helper,
    file_path: str,
    process_num: int,
    success_event: Optional[Any],
    end_loop: int,
    start_time: float,
):
    def _check_success_event():
        if success_event is not None and success_event.is_set():
            return True
        return False

    # ---- infer_python: infer from snippet when missing/incomplete ----
    def infer_python(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {"run_complete": True}
        llm_eval = dict(state.get("llm_eval", {}))
        if "python_version" in llm_eval and "python_modules" in llm_eval and llm_eval["python_modules"]:
            return {"llm_eval": llm_eval}
        try:
            inferred = executor.evaluate_file(ollama_helper, file_path)
            if "python_version" in inferred:
                inferred["python_version"] = str(inferred["python_version"])
            if isinstance(inferred.get("python_modules"), dict):
                inferred["python_modules"] = list(inferred["python_modules"].keys())
            return {"llm_eval": inferred}
        except Exception:
            # Conservative fallback if inference fails
            if "python_version" not in llm_eval:
                llm_eval["python_version"] = "3.8"
            if "python_modules" not in llm_eval:
                llm_eval["python_modules"] = []
            return {"llm_eval": llm_eval}

    # ---- parse_imports: AST first, simple scan fallback ----
    def parse_imports(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {"run_complete": True}
        imports = executor.deps.extract_imports_ast(file_path)
        if not imports:
            imports = executor.deps.find_word_in_file(file_path, "import", [])
        return {"parsed_imports": imports}

    # ---- normalize_modules: canonicalize package names and merge inferred+parsed ----
    def normalize_modules(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {"run_complete": True}
        llm_eval = dict(state.get("llm_eval", {}))
        parsed_imports = state.get("parsed_imports", [])
        inferred_modules = llm_eval.get("python_modules", [])
        if isinstance(inferred_modules, dict):
            inferred_modules = list(inferred_modules.keys())
        merged = parsed_imports + inferred_modules
        normalized = executor.pypi.check_module_name(merged)
        llm_eval["python_modules"] = normalized
        if "python_version" not in llm_eval:
            llm_eval["python_version"] = "3.8"
        return {"llm_eval": llm_eval, "normalized_modules": normalized}

    # ---- query_py1: PyPI query + version files + docker_helper + log file init ----
    def query_py1(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {"run_complete": True}
        llm_eval = state["llm_eval"]
        modified_modules, python_version = executor.pypi.get_module_specifics(llm_eval)
        llm_eval = dict(llm_eval)
        llm_eval["python_modules"] = modified_modules
        llm_eval["python_version"] = python_version
        docker_helper = DockerHelper(logging=True)
        project_dir, dir_name, project_file = docker_helper.get_project_dir(file_path)
        file_to_open = f"{project_dir}/output_data_{python_version}.yml"
        with open(file_to_open, "w") as f:
            f.write("---\n")
            f.write(f"python_version: {python_version}\n")
            f.write(f"start_time: {start_time}\n")
            f.write("iterations:\n")
        return {
            "llm_eval": llm_eval,
            "docker_helper": docker_helper,
            "file_to_open": file_to_open,
            "project_dir": project_dir,
            "dir_name": dir_name,
            "project_file": project_file,
            "strategy": "build",
        }

    # ---- select_versions: LLM picks version per module (RAG) ----
    def select_versions(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {"run_complete": True}
        llm_eval = state["llm_eval"]
        module_versions = ollama_helper.get_module_versions(llm_eval)
        llm_eval = dict(llm_eval)
        llm_eval["python_modules"] = module_versions
        return {"llm_eval": llm_eval}

    # ---- generate_dockerfiles: write Dockerfile ----
    def generate_dockerfiles(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {"run_complete": True}
        docker_helper = state["docker_helper"]
        llm_eval = state["llm_eval"]
        docker_helper.create_dockerfile(llm_eval, file_path)
        return {}

    # ---- docker_build: build image; on failure classify (process_error) ----
    def docker_build(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {"run_complete": True}
        docker_helper = state["docker_helper"]
        llm_eval = state["llm_eval"]
        error_handler = state["error_handler"]
        passed, docker_output = docker_helper.build_dockerfile(file_path)
        output, error_type = None, "Unknown"
        if not passed:
            output, error_type = ollama_helper.process_error(docker_output, error_handler, llm_eval)
        return {
            "build_complete": passed,
            "docker_output": docker_output or "",
            "output": output,
            "error_type": error_type or "Unknown",
        }

    # ---- docker_run: run container, capture output ----
    def docker_run(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {"run_complete": True}
        docker_helper = state["docker_helper"]
        docker_output = docker_helper.run_container_test()
        print(docker_output)
        return {"docker_output": docker_output}

    # ---- classify_errors: set output/error_type from docker_output (for run path; build path already has them) ----
    def classify_errors(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {"run_complete": True}
        # From docker_build failure we already have output, error_type. From docker_run we must call process_error.
        if state.get("build_complete") is True and state.get("docker_output"):
            error_handler = state["error_handler"]
            llm_eval = state["llm_eval"]
            output, error_type = ollama_helper.process_error(
                state["docker_output"], error_handler, llm_eval
            )
            return {"output": output, "error_type": error_type or "Unknown"}
        return {}

    # ---- resolve_error: apply fix (naughty_bois, update_llm_eval, shuffle/pop); used after build or run failure ----
    def resolve_error(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {"run_complete": True}
        llm_eval = state["llm_eval"].copy()
        error_handler = dict(state["error_handler"])
        output = state.get("output")
        error_type = state.get("error_type", "Unknown")
        docker_output = state.get("docker_output", "")
        loop = state.get("loop", 1)
        if error_type not in error_handler:
            error_handler[error_type] = 0

        run_complete = False
        # Build failure path: same as old handle_build_error
        if not state.get("build_complete"):
            error_handler = executor.naughty_bois(output, error_handler, error_type, llm_eval)
            llm_eval = executor.update_llm_eval(output, llm_eval)
            if error_type == "ImportError" and "returned a non-zero code: 1" in docker_output:
                zero_code_module = ollama_helper.non_zero_error(docker_output)
                if output and output.get("module"):
                    llm_eval = executor.shuffle_modules(output["module"], zero_code_module, llm_eval)
            if error_type == "NonZeroCode" and "PATH environment" in docker_output and output and output.get("module"):
                llm_eval["python_modules"].pop(output["module"], None)
            loop_next = executor.end_test(
                state["file_to_open"], llm_eval, state["docker_helper"],
                error_type, docker_output, loop, False
            )
            if loop_next is None:
                loop_next = loop + 1
            return {
                "llm_eval": llm_eval,
                "error_handler": error_handler,
                "loop": loop_next,
                "build_complete": False,
                "strategy": "retry_build",
            }

        # Run failure path: same as old handle_run_error
        if "ImportError" in error_type:
            if "DJANGO_SETTINGS_MODULE is undefined" in docker_output:
                run_complete = True
                if success_event is not None:
                    success_event.set()
                llm_eval = executor.update_llm_eval(output, llm_eval)
            else:
                error_handler = executor.naughty_bois(output, error_handler, error_type, llm_eval)
                llm_eval = executor.update_llm_eval(output, llm_eval)
        elif "VersionNotFound" in error_type:
            error_handler = executor.naughty_bois(output, error_handler, error_type, llm_eval)
            llm_eval = executor.update_llm_eval(output, llm_eval)
        elif "DependencyConflict" in error_type:
            error_handler = executor.naughty_bois(output, error_handler, error_type, llm_eval)
            llm_eval = executor.update_llm_eval(output, llm_eval)
        elif "ModuleNotFound" in error_type:
            error_handler = executor.naughty_bois(output, error_handler, error_type, llm_eval)
            llm_eval = executor.update_llm_eval(output, llm_eval)
        elif "AttributeError" in error_type:
            error_handler = executor.naughty_bois(output, error_handler, error_type, llm_eval)
            llm_eval = executor.update_llm_eval(output, llm_eval)
        elif "InvalidVersion" in error_type:
            error_handler = executor.naughty_bois(output, error_handler, error_type, llm_eval)
            llm_eval = executor.update_llm_eval(output, llm_eval)
        elif "NonZeroCode" in error_type:
            error_handler = executor.naughty_bois(output, error_handler, error_type, llm_eval)
            if output and output.get("module") and output["module"] in llm_eval.get("python_modules", {}):
                llm_eval["python_modules"].pop(output["module"])
        elif "SyntaxError" in error_type:
            error_handler = executor.naughty_bois(output, error_handler, error_type, llm_eval)
            llm_eval = executor.update_llm_eval(output, llm_eval)
        elif "NameError" in error_type:
            run_complete = True
            if success_event is not None:
                success_event.set()
            error_handler = executor.naughty_bois(output, error_handler, error_type, llm_eval)
            llm_eval = executor.update_llm_eval(output, llm_eval)
        elif "None" in error_type:
            run_complete = True
            if success_event is not None:
                success_event.set()
            llm_eval = executor.update_llm_eval(None, llm_eval)

        loop_next = loop + 1
        if not run_complete:
            loop_next = executor.end_test(
                state["file_to_open"], llm_eval, state["docker_helper"],
                error_type, docker_output, loop, False
            )
            if loop_next is None:
                loop_next = loop + 1

        return {
            "llm_eval": llm_eval,
            "error_handler": error_handler,
            "loop": loop_next,
            "run_complete": run_complete,
            "build_complete": False if not run_complete else state.get("build_complete", False),
            "strategy": "finalize" if run_complete else "retry_build",
        }

    # ---- decide_strategy: policy-based routing ----
    def decide_strategy(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {"strategy": "finalize"}
        if state.get("run_complete"):
            return {"strategy": "finalize"}
        loop = state.get("loop", 1)
        if loop > end_loop:
            return {"strategy": "finalize"}
        error_type = state.get("error_type", "Unknown")
        output = state.get("output")
        module = output.get("module") if isinstance(output, dict) else ""
        err_sig = f"{error_type}:{module}"
        prev_sig = state.get("last_error_signature", "")
        stagnation = state.get("stagnation_count", 0)
        stagnation = (stagnation + 1) if err_sig == prev_sig else 0

        # If we keep hitting the same error/module, force a refresh of available versions
        if stagnation >= 2:
            strategy = "requery"
        elif error_type in ("DependencyConflict", "VersionNotFound"):
            strategy = "requery"
        else:
            strategy = "retry_build"

        return {
            "strategy": strategy,
            "last_error_signature": err_sig,
            "stagnation_count": stagnation,
        }

    # ---- finalize: wrap up (caller does end_test after graph; this node just marks done) ----
    def finalize(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {}
        return {"run_complete": True}

    # ---- Routing ----
    def after_docker_build(state: ResolutionState) -> Literal["docker_run", "classify_errors"]:
        if _check_success_event():
            return "docker_run"
        if state.get("build_complete"):
            return "docker_run"
        return "classify_errors"

    def after_resolve_error(state: ResolutionState) -> Literal["decide_strategy"]:
        return "decide_strategy"

    def after_decide_strategy(state: ResolutionState) -> Literal["generate_dockerfiles", "query_py1", "finalize"]:
        if _check_success_event():
            return "finalize"
        if state.get("run_complete"):
            return "finalize"
        if state.get("loop", 0) > end_loop:
            return "finalize"
        if state.get("strategy") == "requery":
            return "query_py1"
        if state.get("strategy") == "finalize":
            return "finalize"
        return "generate_dockerfiles"

    def after_finalize(state: ResolutionState) -> Literal["__end__"]:
        return "__end__"

    # ---- Build graph to match workflow diagram ----
    graph = StateGraph(ResolutionState)

    graph.add_node("infer_python", infer_python)
    graph.add_node("parse_imports", parse_imports)
    graph.add_node("normalize_modules", normalize_modules)
    graph.add_node("query_py1", query_py1)
    graph.add_node("select_versions", select_versions)
    graph.add_node("generate_dockerfiles", generate_dockerfiles)
    graph.add_node("docker_build", docker_build)
    graph.add_node("docker_run", docker_run)
    graph.add_node("classify_errors", classify_errors)
    graph.add_node("resolve_error", resolve_error)
    graph.add_node("decide_strategy", decide_strategy)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("infer_python")
    graph.add_edge("infer_python", "parse_imports")
    graph.add_edge("parse_imports", "normalize_modules")
    graph.add_edge("normalize_modules", "query_py1")
    graph.add_edge("query_py1", "select_versions")
    graph.add_edge("select_versions", "generate_dockerfiles")
    graph.add_edge("generate_dockerfiles", "docker_build")
    graph.add_conditional_edges(
        "docker_build", after_docker_build,
        {"docker_run": "docker_run", "classify_errors": "classify_errors"},
    )
    graph.add_edge("docker_run", "classify_errors")
    graph.add_edge("classify_errors", "resolve_error")
    graph.add_conditional_edges("resolve_error", after_resolve_error, {"decide_strategy": "decide_strategy"})
    graph.add_conditional_edges(
        "decide_strategy", after_decide_strategy,
        {"generate_dockerfiles": "generate_dockerfiles", "query_py1": "query_py1", "finalize": "finalize"},
    )
    graph.add_conditional_edges("finalize", after_finalize, {"__end__": END})

    return graph.compile(checkpointer=None)


# ---------------------------------------------------------------------------
# Invoke from test_executor
# ---------------------------------------------------------------------------

def run_resolution_graph(
    executor,
    ollama_helper,
    llm_eval: dict,
    file_path: str,
    process_num: int,
    success_event: Optional[Any],
) -> ResolutionState:
    """Invoke the LangGraph resolution flow. Returns final state."""
    error_handler = default_error_handler()
    initial_state: ResolutionState = {
        "llm_eval": llm_eval,
        "parsed_imports": [],
        "normalized_modules": [],
        "error_handler": error_handler,
        "docker_output": "",
        "output": None,
        "error_type": "Unknown",
        "strategy": "build",
        "last_error_signature": "",
        "stagnation_count": 0,
        "build_complete": False,
        "run_complete": False,
        "loop": 1,
    }
    graph = build_resolution_graph(
        executor=executor,
        ollama_helper=ollama_helper,
        file_path=file_path,
        process_num=process_num,
        success_event=success_event,
        end_loop=executor.end_loop,
        start_time=executor.start_time,
    )
    config = {"recursion_limit": executor.end_loop * 6}
    final_state = graph.invoke(initial_state, config=config)
    return final_state
