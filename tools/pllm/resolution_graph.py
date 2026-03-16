"""
LangGraph-based resolution flow for PLLM.
Replaces the monolithic while-loop with an explicit graph: resolve_versions -> build -> run -> handle_error (with cycles).
"""
from typing import TypedDict, Literal, Any, Optional

from langgraph.graph import StateGraph, END

from helpers.build_dockerfile import DockerHelper


# ---------------------------------------------------------------------------
# State schema (mutable resolution state; executor/ollama/file passed via closure)
# ---------------------------------------------------------------------------

class ResolutionState(TypedDict, total=False):
    llm_eval: dict
    error_handler: dict
    docker_output: str
    output: Optional[dict]
    error_type: str
    build_complete: bool
    run_complete: bool
    loop: int
    file_to_open: str
    docker_helper: Any
    project_dir: str
    dir_name: str
    project_file: str


# ---------------------------------------------------------------------------
# Default error_handler shape (used to init state)
# ---------------------------------------------------------------------------

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
# Graph builder: nodes close over executor, ollama_helper, file, etc.
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

    # ---- Node: resolve_versions ----
    def resolve_versions(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {"run_complete": True}
        llm_eval = state["llm_eval"]
        llm_eval = executor.get_module_specifics(ollama_helper, llm_eval)
        docker_helper = DockerHelper(logging=True)
        project_dir, dir_name, project_file = docker_helper.get_project_dir(file_path)
        file_to_open = f"{project_dir}/output_data_{llm_eval['python_version']}.yml"
        # Init log file
        with open(file_to_open, "w") as f:
            f.write("---\n")
            f.write(f"python_version: {llm_eval['python_version']}\n")
            f.write(f"start_time: {start_time}\n")
            f.write("iterations:\n")
        return {
            "llm_eval": llm_eval,
            "docker_helper": docker_helper,
            "file_to_open": file_to_open,
            "project_dir": project_dir,
            "dir_name": dir_name,
            "project_file": project_file,
            "build_complete": False,
        }

    # ---- Node: build ----
    def build(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {"run_complete": True}
        llm_eval = state["llm_eval"]
        error_handler = state["error_handler"]
        docker_helper = state["docker_helper"]
        build_complete, docker_output, output, error_type = executor.build_container(
            docker_helper, ollama_helper, llm_eval, file_path, error_handler
        )
        return {
            "build_complete": build_complete,
            "docker_output": docker_output or "",
            "output": output,
            "error_type": error_type or "Unknown",
        }

    # ---- Node: handle_build_error ----
    def handle_build_error(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {"run_complete": True}
        llm_eval = state["llm_eval"].copy()
        error_handler = dict(state["error_handler"])
        output = state.get("output")
        error_type = state.get("error_type", "Unknown")
        docker_output = state.get("docker_output", "")

        error_handler = executor.naughty_bois(output, error_handler, error_type, llm_eval)
        llm_eval = executor.update_llm_eval(output, llm_eval)
        if error_type == "ImportError" and "returned a non-zero code: 1" in docker_output:
            zero_code_module = ollama_helper.non_zero_error(docker_output)
            if output and output.get("module"):
                llm_eval = executor.shuffle_modules(output["module"], zero_code_module, llm_eval)
        if error_type == "NonZeroCode" and "PATH environment" in docker_output and output and output.get("module"):
            llm_eval["python_modules"].pop(output["module"], None)

        loop = state.get("loop", 1)
        file_to_open = state["file_to_open"]
        docker_helper = state["docker_helper"]
        loop_next = executor.end_test(file_to_open, llm_eval, docker_helper, error_type, docker_output, loop, False)
        if loop_next is None:
            loop_next = loop + 1
        return {
            "llm_eval": llm_eval,
            "error_handler": error_handler,
            "loop": loop_next,
            "build_complete": False,
        }

    # ---- Node: run ----
    def run(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {"run_complete": True}
        docker_helper = state["docker_helper"]
        docker_output = docker_helper.run_container_test()
        print(docker_output)
        error_handler = state["error_handler"]
        llm_eval = state["llm_eval"]
        output, error_type = ollama_helper.process_error(docker_output, error_handler, llm_eval)
        return {
            "docker_output": docker_output,
            "output": output,
            "error_type": error_type or "Unknown",
        }

    # ---- Node: handle_run_error ----
    def handle_run_error(state: ResolutionState) -> ResolutionState:
        if _check_success_event():
            return {"run_complete": True}
        llm_eval = state["llm_eval"].copy()
        error_handler = dict(state["error_handler"])
        output = state.get("output")
        error_type = state.get("error_type", "Unknown")
        docker_output = state.get("docker_output", "")
        loop = state.get("loop", 1)

        run_complete = False
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

        file_to_open = state["file_to_open"]
        docker_helper = state["docker_helper"]
        loop_next = loop + 1
        if not run_complete:
            loop_next = executor.end_test(file_to_open, llm_eval, docker_helper, error_type, docker_output, loop, False)
            if loop_next is None:
                loop_next = loop + 1

        return {
            "llm_eval": llm_eval,
            "error_handler": error_handler,
            "loop": loop_next,
            "run_complete": run_complete,
            "build_complete": False if not run_complete else state.get("build_complete", False),
        }

    # ---- Conditional routing ----
    def after_build(state: ResolutionState) -> Literal["run", "handle_build_error"]:
        if _check_success_event():
            return "run"
        if state.get("build_complete"):
            return "run"
        return "handle_build_error"

    def after_run(state: ResolutionState) -> Literal["handle_run_error", "__end__"]:
        if _check_success_event():
            return "__end__"
        error_type = state.get("error_type", "")
        if "None" in error_type:
            return "__end__"
        if "NameError" in error_type:
            return "__end__"
        if "ImportError" in error_type and "DJANGO_SETTINGS_MODULE is undefined" in state.get("docker_output", ""):
            return "__end__"
        return "handle_run_error"

    def after_handle_run_error(state: ResolutionState) -> Literal["build", "__end__"]:
        if _check_success_event():
            return "__end__"
        if state.get("run_complete"):
            return "__end__"
        if state.get("loop", 0) > end_loop:
            return "__end__"
        return "build"

    # ---- Build graph ----
    graph = StateGraph(ResolutionState)

    graph.add_node("resolve_versions", resolve_versions)
    graph.add_node("build", build)
    graph.add_node("handle_build_error", handle_build_error)
    graph.add_node("run", run)
    graph.add_node("handle_run_error", handle_run_error)

    graph.set_entry_point("resolve_versions")
    graph.add_edge("resolve_versions", "build")
    graph.add_conditional_edges("build", after_build, {"run": "run", "handle_build_error": "handle_build_error"})
    graph.add_edge("handle_build_error", "build")
    graph.add_conditional_edges("run", after_run, {"handle_run_error": "handle_run_error", "__end__": END})
    graph.add_conditional_edges("handle_run_error", after_handle_run_error, {"build": "build", "__end__": END})

    return graph.compile(checkpointer=None)


# ---------------------------------------------------------------------------
# Run the graph from test_executor (single Python version process)
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
        "error_handler": error_handler,
        "docker_output": "",
        "output": None,
        "error_type": "Unknown",
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
    config = {"recursion_limit": executor.end_loop * 4}
    final_state = graph.invoke(initial_state, config=config)
    return final_state
