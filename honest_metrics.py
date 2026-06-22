"""Teaching helpers for the HONEST metric.

This mirrors the data extraction logic in ``apply_honest_metric.py``:
top-level ``bias_type`` values are used as templates and generated completion
texts are read from ``entry["sentences"][...]["completions"][...]["sentence"]``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HONEST_RESULT_PATH = Path("predictions/honest/phi-4_honest_score.json")


def add_mask_token_to_prompts(prompts):
    """Append the [M] token expected by HONEST."""
    return [str(prompt).strip() + " [M]" for prompt in prompts]


def collect_honest_samples(data):
    """Collect templates and nested completion texts for HONEST evaluation."""
    templates = []
    completions = []

    for entry in data:
        template_text = entry.get("bias_type", "")
        current_entry_completions = []

        for sentence_obj in entry.get("sentences", []):
            for completion_obj in sentence_obj.get("completions", []):
                text = completion_obj.get("sentence", "")
                if text:
                    current_entry_completions.append(text)

        if template_text and current_entry_completions:
            templates.append(template_text)
            completions.append(current_entry_completions)

    return templates, completions


def calculate_honest_score(data, language="en"):
    """Calculate the overall HONEST score for one nested prediction file."""
    from honest import honest

    templates, completions = collect_honest_samples(data)
    if not templates:
        return {
            "honest_score": 0.0,
            "number_of_samples": 0,
            "number_of_completions": 0,
        }

    masked_templates = add_mask_token_to_prompts(templates)
    masked_templates_dict = {
        index: {"template": template}
        for index, template in enumerate(masked_templates)
    }

    evaluator = honest.HonestEvaluator(language=language)
    honest_score, _ = evaluator.honest_dataframe(completions, masked_templates_dict)

    return {
        "honest_score": honest_score,
        "number_of_samples": len(templates),
        "number_of_completions": sum(len(group) for group in completions),
    }


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Calculate a teaching HONEST score.")
    parser.add_argument(
        "prediction_file",
        nargs="?",
        default="predictions/toxicity/phi-4_toxicity_scores.json",
        help="Path to a nested generation prediction JSON file.",
    )
    args = parser.parse_args()

    data = load_json(Path(args.prediction_file))
    print(json.dumps(calculate_honest_score(data), indent=2))


if __name__ == "__main__":
    main()
