#runs one of the four agents over the held-out procedures and saves the picked traces
#we use this to manually compare the methods later

#  python run_eval.py --method llama_bare
#  python run_eval.py --method llama_actions
#  python run_eval.py --method ensemble        --alpha 0.5
#  python run_eval.py --method agentic_ensemble --alpha 0.5 --tool_threshold 0.85 --tool_margin 0.2
#
#each loads its own model so run them sequentially

import argparse
import json
from pathlib import Path

from runner import load_cases, run_inference
from agents import LlamaBareAgent, LlamaActionsAgent, EnsemblePlannerAgent, AgenticEnsembleAgent


_HERE = Path(__file__).parent
_ROOT = _HERE.parent
DEFAULT_HELD_OUT = _HERE / "held_out.json"
DEFAULT_PREDICTIONS = _ROOT / "Extraction_results" / "extraction_predictions.json"
DEFAULT_OUTPUT_DIR = _HERE / "results"


def make_agent(method: str, alpha: float = 0.5, llm_temp: float = 1.0,
               tool_threshold: float = 0.85, tool_margin: float = 0.2):
    if method == "llama_bare":
        return LlamaBareAgent()
    if method == "llama_actions":
        return LlamaActionsAgent()
    if method == "ensemble":
        return EnsemblePlannerAgent(alpha=alpha, llm_temp=llm_temp)
    if method == "agentic_ensemble":
        return AgenticEnsembleAgent(alpha=alpha, llm_temp=llm_temp,
                                    tool_threshold=tool_threshold, tool_margin=tool_margin)
    raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a planner over the held-out procedures and save the picked traces."
    )
    parser.add_argument("--method", required=True,
                        choices=["llama_bare", "llama_actions", "ensemble", "agentic_ensemble"])
    parser.add_argument("--held_out", type=Path, default=DEFAULT_HELD_OUT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0,
                        help="Only run the first N procedures (0 = all)")
    #ensemble + agentic_ensemble knobs (ignored for the other methods)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Ensemble blend weight: 1.0=PRM only, 0.0=base Llama only (default 0.5)")
    parser.add_argument("--llm_temp", type=float, default=1.0,
                        help="Temperature applied to base Llama distribution before blending (default 1.0)")
    #agentic_ensemble-only — gate the graph tool call
    parser.add_argument("--tool_threshold", type=float, default=0.85,
                        help="Tool fires when top blended score is below this (default 0.85)")
    parser.add_argument("--tool_margin", type=float, default=0.2,
                        help="Tool fires when margin to runner-up is below this (default 0.2)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.held_out, args.predictions)
    if args.limit:
        cases = cases[: args.limit]

    give_candidates = args.method != "llama_bare"
    agent = make_agent(args.method, alpha=args.alpha, llm_temp=args.llm_temp,
                       tool_threshold=args.tool_threshold, tool_margin=args.tool_margin)

    print(f"\nRunning method={args.method} on {len(cases)} procedures "
          f"(give_candidates={give_candidates})")
    if args.method in ("ensemble", "agentic_ensemble"):
        print(f"  alpha={args.alpha}  llm_temp={args.llm_temp}")
    if args.method == "agentic_ensemble":
        print(f"  tool_threshold={args.tool_threshold}  tool_margin={args.tool_margin}")
    print()

    traces = run_inference(cases, agent, give_candidates=give_candidates)

    #include alpha (and tool gates for method 4) in the filename so sweeps don't overwrite
    if args.method == "ensemble":
        suffix = f"_alpha{args.alpha:.2f}"
    elif args.method == "agentic_ensemble":
        suffix = f"_alpha{args.alpha:.2f}_t{args.tool_threshold:.2f}_m{args.tool_margin:.2f}"
    else:
        suffix = ""
    out_path = args.output_dir / f"inference_{args.method}{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(traces)} traces to {out_path}")


if __name__ == "__main__":
    main()
