#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from typing import List

from piper_phonemize import get_espeak_map


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=Path,
        default=Path("data/tokens_elise.txt"),
        help="Path to the dict that maps the text tokens to IDs",
    )
    return parser.parse_args()


def get_token2id(args):
    """Get a dict that maps token to IDs for English-only Elise dataset."""
    # Get espeak tokens for English
    all_tokens = get_espeak_map()  # token: [token_id]
    all_tokens = {token: token_id[0] for token, token_id in all_tokens.items()}
    # sort by token_id
    all_tokens = sorted(all_tokens.items(), key=lambda x: x[1])

    with open(args.tokens, "w", encoding="utf-8") as f:
        for token, token_id in all_tokens:
            f.write(f"{token}\t{token_id}\n")
    
    print(f"Generated token file with {len(all_tokens)} tokens")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    get_token2id(args)
