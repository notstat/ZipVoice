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
    
    # Load dataset in streaming mode to avoid memory issues
    logging.info("Loading Elise dataset in streaming mode...")
    dataset = load_dataset(
        "setfunctionenvironment/provoic3ewy", 
        split="train", 
        streaming=True,  # This is crucial for large datasets
        token=hf_token
    )
    
    # Create a temporary directory to store audio files
    temp_dir = Path("data/elise_audio")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Process in batches and save periodically
    batch_size = 1000
    cuts = []
    all_cuts = []
    
    for idx, example in enumerate(dataset):
        if idx % 100 == 0:
            logging.info(f"Processing example {idx}...")
            
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
        
        # Save cuts in batches to avoid memory issues
        if len(cuts) >= batch_size:
            all_cuts.extend(cuts)
            cuts = []
            logging.info(f"Processed {len(all_cuts)} cuts so far...")
    
    # Add any remaining cuts
    if cuts:
        all_cuts.extend(cuts)
    
    logging.info(f"Total cuts processed: {len(all_cuts)}")
    
    # Create CutSet
    cut_set = CutSet.from_cuts(all_cuts)
    
    # Split into train/dev (95%/5%)
    train_size = int(0.95 * len(cut_set))
    train_cuts = CutSet.from_cuts(all_cuts[:train_size])
    dev_cuts = CutSet.from_cuts(all_cuts[train_size:])
    
    # Save manifests
    train_cuts.to_file(output_dir / "elise_cuts_train.jsonl.gz")
    dev_cuts.to_file(output_dir / "elise_cuts_dev.jsonl.gz")
    
    logging.info(f"Saved {len(train_cuts)} training cuts and {len(dev_cuts)} dev cuts")
    logging.info(f"Audio files saved in {temp_dir}")

if __name__ == "__main__":
    # Clear any dataset cache that might be causing issues
    import shutil
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    elise_cache = cache_dir / "setfunctionenvironment___provoic3ewy"
    if elise_cache.exists():
        logging.info(f"Clearing cache at {elise_cache}")
        shutil.rmtree(elise_cache)
    
    prepare_elise_manifests(Path("data/manifests"))
