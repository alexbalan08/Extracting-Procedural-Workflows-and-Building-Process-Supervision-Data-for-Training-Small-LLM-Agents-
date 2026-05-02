#runs one of the three agents over the held-out procedures and saves the picked traces
#we use this to manually compare the methods later

#  python run_eval.py --method llama_bare
#  python run_eval.py --method llama_actions
#  python run_eval.py --method ensemble --alpha 0.5
#
#each loads its own model so run them sequentially

import argparse
import json
from pathlib import Path

from runner import load_cases, run_inference
from agents import LlamaBareAgent, LlamaActionsAgent, EnsemblePlannerAgent


_HERE = Path(__file__).parent
_ROOT = _HERE.parent
DEFAULT_HELD_OUT = _HERE / "held_out.json"
DEFAULT_PREDICTIONS = _ROOT / "Extraction_results" / "extraction_predictions.json"
DEFAULT_OUTPUT_DIR = _HERE / "results"


def make_agent(method: str, alpha: float = 0.5, llm_temp: float = 1.0):
    if method == "llama_bare":
        return LlamaBareAgent()
    if method == "llama_actions":
        return LlamaActionsAgent()
    if method == "ensemble":
        return EnsemblePlannerAgent(alpha=alpha, llm_temp=llm_temp)
    raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a planner over the held-out procedures and save the picked traces."
    )
    parser.add_argument("--method", required=True,
                        choices=["llama_bare", "llama_actions", "ensemble"])
    parser.add_argument("--held_out", type=Path, default=DEFAULT_HELD_OUT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0,
                        help="Only run the first N procedures (0 = all)")
    #ensemble-only knobs (ignored for the other methods)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Ensemble blend weight: 1.0=PRM only, 0.0=base Llama only (default 0.5)")
    parser.add_argument("--llm_temp", type=float, default=1.0,
                        help="Temperature applied to base Llama distribution before blending (default 1.0)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.held_out, args.predictions)
    if args.limit:
        cases = cases[: args.limit]

    give_candidates = args.method != "llama_bare"
    agent = make_agent(args.method, alpha=args.alpha, llm_temp=args.llm_temp)

    print(f"\nRunning method={args.method} on {len(cases)} procedures "
          f"(give_candidates={give_candidates})")
    if args.method == "ensemble":
        print(f"  alpha={args.alpha}  llm_temp={args.llm_temp}")
    print()

    traces = run_inference(cases, agent, give_candidates=give_candidates)

    #include alpha in the filename so you can sweep without overwriting
    suffix = f"_alpha{args.alpha:.2f}" if args.method == "ensemble" else ""
    out_path = args.output_dir / f"inference_{args.method}{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(traces)} traces to {out_path}")


if __name__ == "__main__":
    main()
