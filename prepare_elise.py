#!/usr/bin/env python3
import os
from pathlib import Path
from datasets import load_dataset
from lhotse import CutSet, Recording, SupervisionSegment, MonoCut
from lhotse.audio import AudioSource
import soundfile as sf
import numpy as np
import tempfile
import logging

logging.basicConfig(level=logging.INFO)

def prepare_elise_manifests(output_dir: Path):
    """Prepare Lhotse manifests for Elise dataset"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get HF token from environment variable
    hf_token = os.environ.get("hf_token")
    if not hf_token:
        raise ValueError("Please set the 'hf_token' environment variable")
    
    # Load dataset - force redownload if there are cache issues
    logging.info("Loading Elise dataset...")
    try:
        # Try loading with streaming first to avoid cache issues
        dataset = load_dataset("setfunctionenvironment/provoic3ewy", split="train", streaming=False, token=hf_token)
        # Convert to list if it's an iterable dataset
        if hasattr(dataset, '__iter__') and not hasattr(dataset, '__len__'):
            dataset = list(dataset)
    except Exception as e:
        logging.warning(f"Failed to load with default method: {e}")
        # Force download without cache
        dataset = load_dataset("setfunctionenvironment/provoic3ewy", split="train", cache_dir=None, download_mode="force_redownload", token=hf_token)

    cuts = []
    
    # Create a temporary directory to store audio files
    # This is needed because Lhotse's compute_fbank expects file paths
    temp_dir = Path("data/elise_audio")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle both dataset formats
    if hasattr(dataset, '__len__'):
        total_examples = len(dataset)
    else:
        # If it's a generator, we don't know the total length
        total_examples = None
        dataset = list(dataset)
        total_examples = len(dataset)
    
    for idx, example in enumerate(dataset):
        if idx % 100 == 0:
            if total_examples:
                logging.info(f"Processing {idx}/{total_examples}")
            else:
                logging.info(f"Processing {idx}...")
            
        # Get audio data
        audio_data = example["audio"]
        
        # Handle both dict format (with 'array' key) and direct array format
        if isinstance(audio_data, dict):
            audio_array = audio_data["array"]
            sampling_rate = audio_data["sampling_rate"]
        else:
            # If it's directly a wav file or array
            audio_array = audio_data
            sampling_rate = 24000  # Default, adjust as needed
        
        # Ensure audio_array is a numpy array
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)
        
        # Ensure audio is 1D
        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze()
        
        # Create unique ID
        audio_id = f"elise_{idx:06d}"
        
        # Save audio to file for Lhotse processing
        audio_path = temp_dir / f"{audio_id}.wav"
        sf.write(str(audio_path), audio_array, sampling_rate)
        
        # Create Recording object
        recording = Recording(
            id=audio_id,
            sources=[
                AudioSource(
                    type="file",
                    channels=[0],
                    source=str(audio_path)
                )
            ],
            sampling_rate=sampling_rate,
            num_samples=len(audio_array),
            duration=len(audio_array) / sampling_rate,
            channel_ids=[0]
        )
        
        # Create SupervisionSegment
        supervision = SupervisionSegment(
            id=audio_id,
            recording_id=audio_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            text=example["text"],
            language="en",
            speaker=example.get("speaker", "default_speaker")
        )
        
        # Create Cut
        cut = MonoCut(
            id=audio_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            recording=recording,
            supervisions=[supervision],
        )
        cuts.append(cut)
    
    # Create CutSet
    cut_set = CutSet.from_cuts(cuts)
    
    # Split into train/dev (95%/5%)
    train_size = int(0.95 * len(cut_set))
    train_cuts = CutSet.from_cuts(cuts[:train_size])
    dev_cuts = CutSet.from_cuts(cuts[train_size:])
    
    # Save manifests
    train_cuts.to_file(output_dir / "elise_cuts_train.jsonl.gz")
    dev_cuts.to_file(output_dir / "elise_cuts_dev.jsonl.gz")
    
    logging.info(f"Saved {len(train_cuts)} training cuts and {len(dev_cuts)} dev cuts")
    logging.info(f"Audio files saved in {temp_dir}")

if __name__ == "__main__":
    # Clear any dataset cache that might be causing issues
    import shutil
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    elise_cache = cache_dir / "MrDragonFox___elise"
    if elise_cache.exists():
        logging.info(f"Clearing cache at {elise_cache}")
        shutil.rmtree(elise_cache)
    
    prepare_elise_manifests(Path("data/manifests"))
