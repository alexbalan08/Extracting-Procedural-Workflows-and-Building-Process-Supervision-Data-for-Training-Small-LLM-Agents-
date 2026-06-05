#runs one of the four agents over the held-out procedures and saves the picked traces
#we use this to manually compare the methods later



#please run any of these the default parameters i included them already 
#  python run_eval.py --method llama_bare
#  python run_eval.py --method llama_actions
#  python run_eval.py --method ensemble        --alpha 0.9
#  python run_eval.py --method agentic_ensemble --alpha 0.9 --tool_threshold 0.45 --tool_margin 0.2




#both modes validate against gold execution_states 
#graph predicted agent sees predicted graph and it measures real deployment quality
#graph gold agent sees gold graph andmeasure what theoretically the agent could do
#gap between gold and predicted = cost of extraction errors

#run those
#  python run_eval.py --method ensemble --alpha 0.9 --graph predicted
#  python run_eval.py --method ensemble --alpha 0.9 --graph gold


#each loads its own model so run them sequentially
#please use the parematers from here for best results since i carefully tested and they are the best in practice

import argparse
import json
from pathlib import Path

from runner import load_cases, run_inference
from agents import (
    LlamaBareAgent, LlamaActionsAgent,
    EnsemblePlannerAgent, AgenticEnsembleAgent,
    OpenAIBareAgent, OpenAIActionsAgent,
)


_HERE = Path(__file__).parent
_ROOT = _HERE.parent
DEFAULT_HELD_OUT = _HERE / "held_out.json"
DEFAULT_PREDICTIONS = _ROOT / "Extraction_results" / "extraction_predictions.json"
DEFAULT_OUTPUT_DIR = _HERE / "results"
DEFAULT_PRM_ADAPTER = _ROOT / "PRM" / "trained_model"


def make_agent(method: str, alpha: float = 0.5, llm_temp: float = 1.0,
               tool_threshold: float = 0.85, tool_margin: float = 0.2,
               prm_adapter: Path = DEFAULT_PRM_ADAPTER,
               openai_model: str = "gpt-5.4-mini"):
    if method == "llama_bare":
        return LlamaBareAgent()
    if method == "llama_actions":
        return LlamaActionsAgent()
    if method == "ensemble":
        return EnsemblePlannerAgent(adapter_path=prm_adapter, alpha=alpha, llm_temp=llm_temp)
    if method == "agentic_ensemble":
        return AgenticEnsembleAgent(adapter_path=prm_adapter, alpha=alpha, llm_temp=llm_temp,
                                    tool_threshold=tool_threshold, tool_margin=tool_margin)
    if method == "openai_bare":
        return OpenAIBareAgent(model=openai_model)
    if method == "openai_actions":
        return OpenAIActionsAgent(model=openai_model)
    raise ValueError(f"Unknown method: {method}")


def _prm_tag(adapter_path: Path) -> str:
    name = adapter_path.name
    if name == "trained_model":
        return ""
    if name.startswith("trained_model_"):
        return "_" + name[len("trained_model_"):]
    return "_" + name


def main():
    parser = argparse.ArgumentParser(
        description="Run a planner over the held-out procedures and save the picked traces."
    )
    parser.add_argument("--method", required=True,
                        choices=["llama_bare", "llama_actions", "ensemble", "agentic_ensemble",
                                 "openai_bare", "openai_actions"])
    parser.add_argument("--held_out", type=Path, default=DEFAULT_HELD_OUT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0,
                        help="Only run the first N procedures (0 = all)")
    parser.add_argument("--graph", choices=["gold", "predicted"], default="predicted",
                        help="Evaluation mode. Both validate against the gold graph. "
                             "gold = agent also SEES the gold graph — measures agent ceiling. "
                             "predicted = agent sees the predicted graph — measures real deployment. "
                             "gap between gold and predicted = cost of extraction errors.")
   

    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Ensemble blend weight: 1.0=PRM only, 0.0=base Llama only (default 0.5)")
    parser.add_argument("--llm_temp", type=float, default=1.0,
                        help="Temperature applied to base Llama distribution before blending (default 1.0)")
    
    parser.add_argument("--tool_threshold", type=float, default=0.85,
                        help="Tool fires when top blended score is below this (default 0.85)")
    parser.add_argument("--tool_margin", type=float, default=0.2,
                        help="Tool fires when margin to runner-up is below this (default 0.2)")
    parser.add_argument("--prm_adapter", type=Path, default=DEFAULT_PRM_ADAPTER,
                        help="Path to the PRM LoRA adapter folder. Use PRM/trained_model_small "
                             "to compare the dedup-data PRM against the default model.")
    parser.add_argument("--openai_model", type=str, default="gpt-5.4-mini",
                        help="OpenAI model name for --method openai_bare (default gpt-5.4-mini)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.held_out, args.predictions)
    if args.limit:
        cases = cases[: args.limit]

    give_candidates = args.method not in ("llama_bare", "openai_bare")
    
    agent = make_agent(args.method, alpha=args.alpha, llm_temp=args.llm_temp,
                       tool_threshold=args.tool_threshold, tool_margin=args.tool_margin,
                       prm_adapter=args.prm_adapter, openai_model=args.openai_model)

    print(f"\nRunning method={args.method} on {len(cases)} procedures "
          f"(graph={args.graph}, give_candidates={give_candidates})")
    if args.method in ("ensemble", "agentic_ensemble"):
        print(f"  alpha={args.alpha}  llm_temp={args.llm_temp}  prm_adapter={args.prm_adapter}")
    if args.method == "agentic_ensemble":
        print(f"  tool_threshold={args.tool_threshold}  tool_margin={args.tool_margin}")
    print()

    traces = run_inference(cases, agent, give_candidates=give_candidates, mode=args.graph)

    
    suffix = f"_{args.graph}"
    if args.method == "ensemble":
        suffix += f"_alpha{args.alpha:.2f}"
    elif args.method == "agentic_ensemble":
        suffix += f"_alpha{args.alpha:.2f}_t{args.tool_threshold:.2f}_m{args.tool_margin:.2f}"
    #PRM tag only appended for methods that actually use the PRM
    if args.method in ("ensemble", "agentic_ensemble"):
        suffix += _prm_tag(args.prm_adapter)
    #openai model name in the filename so different models don't collide
    if args.method in ("openai_bare", "openai_actions"):
        suffix += f"_{args.openai_model}"
    out_path = args.output_dir / f"inference_{args.method}{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(traces)} traces to {out_path}")


if __name__ == "__main__":
    main()
