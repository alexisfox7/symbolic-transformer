#!/usr/bin/env python3
"""
Generate distractor-set examples from paired A/B stories.

Input format (final_dataset.json):
[
  ["paired_000001_A", "A1", "A2", "A3"],
  ["paired_000001_B", "B1", "B2", "B3"],
  ...
]

Output format (pair_data.json):
[
  {
    "id": "paired_000001_distractor_A_1",
    "target_story": "paired_000001_A",
    "target_sentence": "A3",
    "context_sentences": [A1_or_B1/A2_or_B2 shuffled ... 4 total],
    "distractor_source": "paired_000001_B"
  },
  ...
]
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

def load_dataset(path: str) -> List[List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list.")
    for i, item in enumerate(data):
        if not (isinstance(item, list) and len(item) == 4 and isinstance(item[0], str)):
            raise ValueError(f"Item {i} is not of the form [id, s1, s2, s3]. Got: {item}")
    return data

def group_pairs(data: List[List[str]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns mapping:
      base_id -> {"A": [s1,s2,s3], "B": [s1,s2,s3]}
    where base_id = id without the trailing _A/_B.
    """
    groups: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    for item in data:
        pid = item[0]  # like "paired_000123_A"
        if "_" not in pid:
            raise ValueError(f"Story id must contain suffix _A or _B: {pid}")
        base, suffix = pid.rsplit("_", 1)
        if suffix not in ("A", "B"):
            raise ValueError(f"Story id must end with _A or _B: {pid}")
        groups[base][suffix] = item[1:]
    return groups

def validate_complete_pairs(groups: Dict[str, Dict[str, List[str]]]) -> Tuple[List[str], List[str]]:
    complete = []
    incomplete = []
    for base, pair in groups.items():
        if "A" in pair and "B" in pair and len(pair["A"]) == 3 and len(pair["B"]) == 3:
            complete.append(base)
        else:
            incomplete.append(base)
    return complete, incomplete

def make_variants_for_target(
    base: str,
    pair: Dict[str, List[str]],
    target_label: str,
    n_variants: int,
    rng: random.Random
) -> List[dict]:
    """
    Build N variants for target story {A|B} with the other as distractor.
    context_sentences = shuffled [T1, T2, C1, C2]
    target_sentence   = T3
    """
    if target_label == "A":
        T1, T2, T3 = pair["A"]
        C1, C2, _C3 = pair["B"]
        tgt_story = f"{base}_A"
        distractor_src = f"{base}_B"
    else:
        T1, T2, T3 = pair["B"]
        C1, C2, _C3 = pair["A"]
        tgt_story = f"{base}_B"
        distractor_src = f"{base}_A"

    out = []
    for k in range(n_variants):
        ctx = [T1, T2, C1, C2]
        rng.shuffle(ctx)
        out.append({
            "id": f"{base}_distractor_{target_label}_{k+1}",
            "target_story": tgt_story,
            "target_sentence": T3,
            "context_sentences": ctx,
            "distractor_source": distractor_src
        })
    return out

def build_distractors(
    groups: Dict[str, Dict[str, List[str]]],
    n_variants_per_target: int,
    seed: int
) -> Tuple[List[dict], dict]:
    rng = random.Random(seed)
    examples: List[dict] = []
    complete_bases, incomplete_bases = validate_complete_pairs(groups)

    for base in complete_bases:
        pair = groups[base]
        examples.extend(make_variants_for_target(base, pair, "A", n_variants_per_target, rng))
        examples.extend(make_variants_for_target(base, pair, "B", n_variants_per_target, rng))

    stats = {
        "num_bases": len(groups),
        "complete_pairs": len(complete_bases),
        "incomplete_pairs": len(incomplete_bases),
        "incomplete_bases": incomplete_bases,
        "examples_generated": len(examples),
        "variants_per_target": n_variants_per_target,
        "seed": seed,
    }
    return examples, stats

def main():
    ap = argparse.ArgumentParser(description="Generate distractor-set examples from paired A/B stories.")
    ap.add_argument("--input", default="final_dataset.json", help="Path to input dataset JSON.")
    ap.add_argument("--output", default="pair_data.json", help="Path to write distractor examples JSON.")
    ap.add_argument("--stats", default="pair_data.stats.json", help="Path to write stats JSON.")
    ap.add_argument("--variants", type=int, default=5, help="Variants per target (A and B).")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility.")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    data = load_dataset(args.input)
    groups = group_pairs(data)
    examples, stats = build_distractors(groups, args.variants, args.seed)

    # Save outputs
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(examples)} examples to {args.output}")
    print(f"Wrote stats to {args.stats}")
    print(f"Complete pairs: {stats['complete_pairs']}/{stats['num_bases']}")
    if stats["incomplete_pairs"]:
        print("Warning: some bases missing A or B:", stats["incomplete_bases"])

if __name__ == "__main__":
    main()
