#!/usr/bin/env python3
import argparse
import re
from collections import Counter
from pathlib import Path
from lhotse import load_manifest_lazy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=Path,
        default=Path("data/tokens_elise.txt"),
        help="Path to the dict that maps the text tokens to IDs",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/fbank/elise_cuts_train.jsonl.gz"),
        help="Path to the manifest file",
    )
    return parser.parse_args()

def prepare_tokens(manifest_file, token_file):
    counter = Counter()
    manifest = load_manifest_lazy(manifest_file)
    
    for cut in manifest:
        text = cut.supervisions[0].text
        # Normalize text
        text = re.sub(r"\s+", " ", text)
        text = text.lower()  # Convert to lowercase
        counter.update(text)
    
    unique_chars = set(counter.keys())
    
    # Remove underscore if present
    if "_" in unique_chars:
        unique_chars.remove("_")
    
    # Sort by frequency
    sorted_chars = sorted(unique_chars, key=lambda char: counter[char], reverse=True)
    
    # Add blank token at the beginning
    result = ["_"] + sorted_chars
    
    # Write to file
    with open(token_file, "w", encoding="utf-8") as f:
        for index, char in enumerate(result):
            f.write(f"{char}\t{index}\n")
    
    print(f"Generated token file with {len(result)} tokens")

if __name__ == "__main__":
    args = get_args()
    prepare_tokens(args.manifest, args.tokens)
