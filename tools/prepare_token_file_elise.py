#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from lhotse import load_manifest_lazy
from piper_phonemize import get_espeak_map
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials
import logging
from typing import List

# This function is copied directly from prepare_token_file_emilia.py
def get_pinyin_tokens(pinyin_file: Path) -> List[str]:
    """Reads a pinyin file and processes it into separate initial/final tokens."""
    phones = set()
    if not pinyin_file.exists():
        logging.warning(f"Pinyin file not found at {pinyin_file}, skipping pinyin tokens.")
        return []
    with open(pinyin_file, "r") as f:
        for line in f:
            x = line.strip()
            initial = to_initials(x, strict=False)
            # Use tone3 style to avoid conflicts with espeak tokens
            finals = to_finals_tone3(x, strict=False, neutral_tone_with_five=True)
            if initial != "":
                # Add a '0' to avoid conflicts with espeak tokens
                phones.add(initial + "0")
            if finals != "":
                phones.add(finals)
    return sorted(list(phones))

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
        default=Path("data/manifests/elise_cuts_train.jsonl.gz"),
        help="Path to the manifest file to scan for tags.",
    )
    # Add the pinyin argument, just like in the emilia script
    parser.add_argument(
        "--pinyin",
        type=Path,
        default=Path("resources/pinyin.txt"),
        help="Path to the all unique pinyin file.",
    )
    return parser.parse_args()

def prepare_tokens(manifest_file: Path, token_file: Path, pinyin_file: Path):
    """
    Builds a vocabulary by combining in order:
    1. Base espeak phonemes with their original IDs.
    2. Pinyin tokens (from emilia script logic).
    3. Custom tags/words from the Elise manifest.
    This creates a single contiguous block of token IDs.
    """
    # 1. Get the base espeak phonemes and their original IDs
    espeak_map = get_espeak_map()
    espeak_tokens = {token: ids[0] for token, ids in espeak_map.items()}

    # 2. Get pinyin tokens and filter out any that already exist in espeak
    pinyin_tokens_list = get_pinyin_tokens(pinyin_file)
    pinyin_tokens = [p for p in pinyin_tokens_list if p not in espeak_tokens]

    # 3. Scan the manifest to find custom tags/words
    logging.info(f"Scanning manifest {manifest_file} for special tags...")
    if not manifest_file.exists():
        logging.error(f"Manifest file not found at {manifest_file}. Please ensure Stage 0 of prepare_elise.sh has run.")
        return

    manifest = load_manifest_lazy(manifest_file)
    angle_tag_pattern = re.compile(r"(<[^>]+>)")
    square_tag_pattern = re.compile(r"\[([^\]]+)\]")
    
    custom_tokens_to_add = {"[", "]"}

    for cut in manifest:
        text = cut.supervisions[0].text
        custom_tokens_to_add.update(angle_tag_pattern.findall(text))
        for match_content in square_tag_pattern.findall(text):
            custom_tokens_to_add.update(re.findall(r'\w+', match_content))
    
    # Filter out any custom tokens that already exist in espeak or pinyin
    existing_tokens = set(espeak_tokens.keys()).union(set(pinyin_tokens))
    new_custom_tokens = sorted([t for t in custom_tokens_to_add if t not in existing_tokens])

    # 4. Write the token file with contiguous IDs
    # Sort espeak tokens by their ID for a deterministic file layout
    sorted_espeak = sorted(espeak_tokens.items(), key=lambda item: item[1])
    
    with open(token_file, "w", encoding="utf-8") as f:
        # Step A: Write all original espeak tokens
        for token, token_id in sorted_espeak:
            f.write(f"{token}\t{token_id}\n")

        # Find the maximum ID from the espeak tokens to start the next block from
        max_id = sorted_espeak[-1][1] if sorted_espeak else -1
        next_id = max_id + 1

        # Step B: Write pinyin tokens, continuing the ID sequence
        logging.info(f"Assigning pinyin tokens starting from ID {next_id}")
        for token in pinyin_tokens:
            f.write(f"{token}\t{next_id}\n")
            next_id += 1

        # Step C: Write new custom tokens, continuing the ID sequence
        logging.info(f"Assigning new custom tokens starting from ID {next_id}")
        for token in new_custom_tokens:
            f.write(f"{token}\t{next_id}\n")
            next_id += 1
    
    logging.info(f"Generated token file '{token_file}' with {next_id} total tokens.")
    if pinyin_tokens:
        logging.info(f"Added {len(pinyin_tokens)} pinyin tokens.")
    if new_custom_tokens:
        logging.info(f"Added {len(new_custom_tokens)} custom Elise tags/words. Examples: {new_custom_tokens[:5]}...")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    prepare_tokens(args.manifest, args.tokens, args.pinyin)
