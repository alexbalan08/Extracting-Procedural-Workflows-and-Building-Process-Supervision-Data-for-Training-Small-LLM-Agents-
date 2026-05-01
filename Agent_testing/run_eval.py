#runs one of the three agents over the held-out procedures and saves the picked traces
#we use this to manually compare the methods later

#  python run_eval.py --method llama_bare      method 1: vanilla llama, no actions list
#  python run_eval.py --method llama_actions   method 2: llama + extracted actions
#  python run_eval.py --method prm             method 3: prm as the picker

#each loads its own model so run them sequentially
#flags: --limit N for a smoke test, --max_steps N to cap rollout length

import argparse
import json
from pathlib import Path

from runner import load_cases, run_inference
from agents import LlamaBareAgent, LlamaActionsAgent, PRMAgent


_HERE = Path(__file__).parent
_ROOT = _HERE.parent
DEFAULT_HELD_OUT = _HERE / "held_out.json"
DEFAULT_PREDICTIONS = _ROOT / "Extraction_results" / "extraction_predictions.json"
DEFAULT_OUTPUT_DIR = _HERE / "results"


def make_agent(method: str):
    if method == "llama_bare":
        return LlamaBareAgent()
    if method == "llama_actions":
        return LlamaActionsAgent()
    if method == "prm":
        return PRMAgent()
    raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a planner over the held-out procedures and save the picked traces."
    )
    parser.add_argument("--method", required=True,
                        choices=["llama_bare", "llama_actions", "prm"])
    parser.add_argument("--held_out", type=Path, default=DEFAULT_HELD_OUT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_steps", type=int, default=20,
                        help="Number of steps to roll the agent forward per procedure")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only run the first N procedures (0 = all)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.held_out, args.predictions)
    if args.limit:
        cases = cases[: args.limit]

    give_candidates = args.method != "llama_bare"
    agent = make_agent(args.method)

    print(f"\nRunning method={args.method} on {len(cases)} procedures "
          f"(give_candidates={give_candidates}, max_steps={args.max_steps})\n")


    traces = run_inference(cases, agent, max_steps=args.max_steps,
                           give_candidates=give_candidates)


    out_path = args.output_dir / f"inference_{args.method}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(traces)} traces to {out_path}")


if __name__ == "__main__":
    main()
