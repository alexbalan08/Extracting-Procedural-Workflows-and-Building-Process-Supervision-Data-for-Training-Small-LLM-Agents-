#runs all 4 agents combinations (for now) on both graph modes (predicted and gold) and we save the tarces
#for validation and compute the scores
#we reload the model so might take a while just chill
#in terms of paramters alpha and for the tool case margin we use the parameters that i described 
#in agents.py as the best ones 


#  python run_all.py
#  python run_all.py --force
#  python run_all.py --limit 5     # quick smoke test

import argparse
import subprocess
from pathlib import Path

_HERE = Path(__file__).parent
RESULTS_DIR = _HERE / "results"

#these are the validated defaults from run_eval.py
#suffix has to match the one run_eval.py builds so we can detect existing outputs
RUNS = [
    ("llama_bare",       [], ""),
    ("llama_actions",    [], ""),
    ("ensemble",         ["--alpha", "0.9"], "_alpha0.90"),
    ("agentic_ensemble", ["--alpha", "0.9", "--tool_threshold", "0.45", "--tool_margin", "0.2"],
                         "_alpha0.90_t0.45_m0.20"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Re-run even if the output file already exists")
    p.add_argument("--limit", type=int, default=0,
                   help="Forwarded to run_eval — only N procedures (0 = all)")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for method, extra, suffix in RUNS:
        for graph in ("predicted", "gold"):
            out = RESULTS_DIR / f"inference_{method}_{graph}{suffix}.json"
            if out.exists() and not args.force:
                print(f"[skip] {out.name} already exists")
                continue
            cmd = ["python", "run_eval.py", "--method", method, "--graph", graph] + extra
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            print(f"\n=== {method} on {graph} ===")
            subprocess.run(cmd, cwd=_HERE, check=True)


if __name__ == "__main__":
    main()
