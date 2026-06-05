#one-button runner: covers EVERY configuration we want in the comparison table.
#  - llama_bare       × {predicted, gold}                       = 2 runs (no PRM)
#  - llama_actions    × {predicted, gold}                       = 2 runs (no PRM)
#  - ensemble         × {predicted, gold} × {big, small} PRM    = 4 runs
#  - agentic_ensemble × {predicted, gold} × {big, small} PRM    = 4 runs
#  total: 12 runs.
#
#just press run — no CLI args. skips any config whose output file already exists,
#so re-running is safe (picks up where it left off after a crash or interrupt).
#
#after it finishes:
#  python evaluate_traces.py    # the full comparison table

import subprocess
import sys
from pathlib import Path


_HERE = Path(__file__).parent
_ROOT = _HERE.parent
RESULTS_DIR = _HERE / "results"
RUN_EVAL = _HERE / "run_eval.py"

BIG_ADAPTER   = _ROOT / "PRM" / "trained_model"
SMALL_ADAPTER = _ROOT / "PRM" / "trained_model_small"

#kept in sync with the canonical config used in the rest of the project
ALPHA          = 0.9
TOOL_THRESHOLD = 0.45
TOOL_MARGIN    = 0.2

#methods that don't use the PRM at all — runs once, no adapter argument
NON_PRM_METHODS = ("llama_bare", "llama_actions")


def _prm_tag(adapter: Path) -> str:
    #mirror run_eval._prm_tag — keeps filenames in sync with the rest of the pipeline
    name = adapter.name
    if name == "trained_model":
        return ""
    if name.startswith("trained_model_"):
        return "_" + name[len("trained_model_"):]
    return "_" + name


def _expected_filename(method: str, graph: str, adapter: Path | None) -> str:
    suffix = f"_{graph}"
    if method == "ensemble":
        suffix += f"_alpha{ALPHA:.2f}"
    elif method == "agentic_ensemble":
        suffix += f"_alpha{ALPHA:.2f}_t{TOOL_THRESHOLD:.2f}_m{TOOL_MARGIN:.2f}"
    if adapter is not None:
        suffix += _prm_tag(adapter)
    return f"inference_{method}{suffix}.json"


def _build_configs():
    configs = []
    for graph in ("predicted", "gold"):
        for method in NON_PRM_METHODS:
            configs.append((method, graph, None))
    for graph in ("predicted", "gold"):
        for adapter in (BIG_ADAPTER, SMALL_ADAPTER):
            configs.append(("ensemble",         graph, adapter))
            configs.append(("agentic_ensemble", graph, adapter))
    return configs


def main():
    configs = _build_configs()
    print(f"Running {len(configs)} configurations (big PRM vs small PRM, plus the non-PRM baselines).")
    print("Skipping any whose output file already exists.\n")

    n_failed = 0
    for i, (method, graph, adapter) in enumerate(configs, 1):
        if adapter is None:
            prm_label = "-"
        else:
            prm_label = "small" if "_small" in adapter.name else "big"

        target_name = _expected_filename(method, graph, adapter)
        target_path = RESULTS_DIR / target_name

        header = f"[{i:2d}/{len(configs)}] {method:17s} | {graph:9s} | PRM={prm_label}"
        if target_path.exists():
            print(f"{header}  ->  SKIP (already have {target_name})")
            continue

        print(f"{header}  ->  running ...")

        cmd = [
            sys.executable, str(RUN_EVAL),
            "--method", method,
            "--graph", graph,
        ]
        if method in ("ensemble", "agentic_ensemble"):
            cmd += ["--alpha", str(ALPHA)]
            cmd += ["--prm_adapter", str(adapter)]
        if method == "agentic_ensemble":
            cmd += ["--tool_threshold", str(TOOL_THRESHOLD),
                    "--tool_margin",    str(TOOL_MARGIN)]

        result = subprocess.run(cmd, cwd=str(_HERE))
        if result.returncode != 0:
            #don't kill the whole batch if one config fails (e.g. GPU was momentarily busy).
            #just move on; user can re-run the script later, the skip-existing logic will
            #only re-attempt the configs whose output file is missing.
            print(f"  FAILED with exit code {result.returncode}. Continuing.")
            n_failed += 1

    print(f"\nAll configs attempted. Failed: {n_failed}/{len(configs)}.")
    print("Run `python evaluate_traces.py` to see the comparison table.")


if __name__ == "__main__":
    main()
