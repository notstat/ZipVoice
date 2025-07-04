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
    Builds a vocabulary by combining:
    1. Base espeak phonemes.
    2. Sound/action tags like <laugh> as whole units.
    3. Emotion words from inside [emotion, words] tags.
    4. The bracket characters [ and ] themselves.
    """
    # 1. Start with espeak phonemes and hardcoded bracket tokens
    final_tokens = {"_", "[", "]"}
    final_tokens.update(get_espeak_map().keys())

    # 2. Scan the manifest for special tags
    logging.info(f"Scanning manifest {manifest_file} for special tags...")
    if not manifest_file.exists():
        logging.error(f"Manifest file not found at {manifest_file}. Please ensure Stage 0 of prepare_elise.sh has run.")
        return

    manifest = load_manifest_lazy(manifest_file)
    angle_tag_pattern = re.compile(r"(<[^>]+>)")
    square_tag_pattern = re.compile(r"\[([^\]]+)\]")
    
    found_angle_tags = set()
    found_emotion_words = set()

    for cut in manifest:
        text = cut.supervisions[0].text
        
        # Find all <...> tags and add them as whole tokens
        angle_tags = angle_tag_pattern.findall(text)
        found_angle_tags.update(angle_tags)

        # Find all [...] tags and extract the words inside
        square_tag_matches = square_tag_pattern.findall(text)
        for match_content in square_tag_matches:
            # Extract individual words
            words = re.findall(r'\w+', match_content)
            found_emotion_words.update(words)

    if found_angle_tags:
        logging.info(f"Found {len(found_angle_tags)} unique <...> tags: {sorted(list(found_angle_tags))}")
        final_tokens.update(found_angle_tags)

    if found_emotion_words:
        logging.info(f"Found {len(found_emotion_words)} unique emotion words: {sorted(list(found_emotion_words))}")
        final_tokens.update(found_emotion_words)

    # 3. Create the final token list and write to file
    sorted_tokens = sorted(list(final_tokens))
    if "_" in sorted_tokens:
        sorted_tokens.remove("_")
    
    result = ["_"] + sorted_tokens
    
    with open(token_file, "w", encoding="utf-8") as f:
        for index, token in enumerate(result):
            f.write(f"{token}\t{index}\n")
    
    logging.info(f"Generated token file '{token_file}' with {len(result)} tokens.")

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    prepare_tokens(args.manifest, args.tokens)
