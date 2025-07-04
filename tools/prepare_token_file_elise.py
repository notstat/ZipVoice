#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from lhotse import load_manifest_lazy
from piper_phonemize import get_espeak_map
import logging

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
    return parser.parse_args()

def prepare_tokens(manifest_file: Path, token_file: Path):
    """
    Builds a vocabulary by:
    1. Writing all original espeak phonemes with their original IDs.
    2. Appending any new custom tags with new, sequential IDs starting from 3000.
    """
    # Define a high starting ID for custom tokens to avoid any conflicts.
    CUSTOM_TOKEN_START_ID = 3000

    # 1. Get the base espeak phonemes and their original IDs
    espeak_map = get_espeak_map()
    espeak_tokens = {token: ids[0] for token, ids in espeak_map.items()}

    # 2. Scan the manifest to find custom tags that need to be added
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
        
        # Find all <...> tags
        custom_tokens_to_add.update(angle_tag_pattern.findall(text))

        # Find all [...] tags and extract the words inside
        for match_content in square_tag_pattern.findall(text):
            custom_tokens_to_add.update(re.findall(r'\w+', match_content))
    
    # Filter out any custom tokens that might already be in espeak's vocabulary
    new_tokens = sorted([t for t in custom_tokens_to_add if t not in espeak_tokens])

    # 3. Write the token file, preserving original espeak IDs
    # Sort espeak tokens by their ID for a deterministic file layout
    sorted_espeak = sorted(espeak_tokens.items(), key=lambda item: item[1])
    
    with open(token_file, "w", encoding="utf-8") as f:
        # First, write all the original espeak tokens with their original IDs
        for token, token_id in sorted_espeak:
            f.write(f"{token}\t{token_id}\n")

        # Now, append all the new custom tokens with sequential IDs starting from 3000
        logging.info(f"Assigning new custom tokens starting from ID {CUSTOM_TOKEN_START_ID}")
        next_id = CUSTOM_TOKEN_START_ID
        for token in new_tokens:
            f.write(f"{token}\t{next_id}\n")
            next_id += 1
    
    total_tokens = len(sorted_espeak) + len(new_tokens)
    logging.info(f"Generated token file '{token_file}' with {total_tokens} total tokens.")
    if new_tokens:
        logging.info(f"Added {len(new_tokens)} custom tokens. Examples: {new_tokens[:5]}...")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    prepare_tokens(args.manifest, args.tokens)
