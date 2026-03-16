"""Agentic extraction loop: extract, check then retry with feedback."""

import argparse
import json
from pathlib import Path

from extractor import WorkflowExtractor
from checker import CombinedChecker


class ExtractionAgent:
    #we extract workflows with the extractor, then check them with the checker,
    # and if there are issues we feed them back to the extractor for another attempt.
    # We repeat this loop until we pass 3 attempts

    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        max_attempts: int = 3,
        extractor_temperature: float = 0.1,
        checker_temperature: float = 0.0,
        max_new_tokens: int = 8000,
        checker_max_new_tokens: int = 512,
    ):
        self.max_attempts = max_attempts
        self.extractor = WorkflowExtractor(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=extractor_temperature,
        )
        self.checker = CombinedChecker(
            model=self.extractor.model,
            tokenizer=self.extractor.tokenizer,
            max_new_tokens=checker_max_new_tokens,
            temperature=checker_temperature,
        )

    def run(self, procedure_text: str) -> dict:
        feedback_issues = None
        history = []

        for attempt in range(1, self.max_attempts + 1):
            print(f"[Agent] Attempt {attempt}/{self.max_attempts} — extracting …")
            result = self.extractor.extract(procedure_text, feedback_issues=feedback_issues, attempt=attempt)

            print(f"[Agent] Attempt {attempt}/{self.max_attempts} — checking …")
            check = self.checker.check(result["workflow"], procedure_text)

            history.append({
                "attempt": attempt,
                "reasoning": result["reasoning"],
                "workflow": result["workflow"],
                "is_valid": check.is_valid,
                "issues": check.issues,
            })

            if check.is_valid:
                print(f"[Agent] Valid on attempt {attempt}.")
                break

            print(f"[Agent] {len(check.issues)} issue(s) — retrying.")
            feedback_issues = check.issues

        last = history[-1]
        return {
            "workflow": last["workflow"],
            "reasoning": last["reasoning"],
            "attempts": last["attempt"],
            "is_valid": last["is_valid"],
            "issues": [] if last["is_valid"] else last["issues"],
            "history": history,
        }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=8000)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str)
    group.add_argument("--input_json", type=Path)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--max_records", type=int, default=None, help="Limit number of records for testing")
    args = parser.parse_args()

    agent = ExtractionAgent(model_path=args.model, max_attempts=args.max_attempts,
                            max_new_tokens=args.max_new_tokens)

    if args.text:
        outputs = [agent.run(args.text)]
    else:
        with open(args.input_json, "r", encoding="utf-8") as f:
            content = f.read().strip()
        records = json.loads(content) if content.startswith("[") else [
            json.loads(line) for line in content.splitlines() if line.strip()
        ]
        if args.max_records:
            records = records[:args.max_records]
        outputs = []
        for i, rec in enumerate(records, 1):
            print(f"\n[{i}/{len(records)}] file_index={rec.get('file_index')}")
            out = agent.run(rec["procedure_text"])
            out["file_index"] = rec.get("file_index")
            outputs.append(out)
            
    #i will store this in processed folder from data
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        print(f"Results written to {args.output_json}")
    else:
        print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.argv = [
        "agent.py",
        "--model",       str(project_root / "outputs"),
        "--input_json",  str(project_root / "Data" / "Processed" / "extracted_test.json"),
        "--output_json", str(project_root / "Data" / "Processed" / "model_predictions.json"),
        "--max_records", "10",
    ]
    main()
