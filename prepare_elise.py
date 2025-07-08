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
import random

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
    
    # Create temp directory for intermediate manifest files
    temp_manifest_dir = output_dir / "temp"
    temp_manifest_dir.mkdir(parents=True, exist_ok=True)
    
    # Process in batches and save periodically
    batch_size = 1000
    batch_num = 0
    cuts = []
    total_processed = 0
    
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
            # Save this batch to a temporary file
            batch_file = temp_manifest_dir / f"batch_{batch_num:04d}.jsonl.gz"
            CutSet.from_cuts(cuts).to_file(batch_file)
            
            total_processed += len(cuts)
            logging.info(f"Saved batch {batch_num} with {len(cuts)} cuts. Total processed: {total_processed}")
            
            # Clear the cuts list to free memory
            cuts = []
            batch_num += 1
    
    # Save any remaining cuts
    if cuts:
        batch_file = temp_manifest_dir / f"batch_{batch_num:04d}.jsonl.gz"
        CutSet.from_cuts(cuts).to_file(batch_file)
        total_processed += len(cuts)
        logging.info(f"Saved final batch {batch_num} with {len(cuts)} cuts. Total processed: {total_processed}")
    
    logging.info(f"Total cuts processed: {total_processed}")
    
    # Now merge all batch files into train/dev splits
    logging.info("Merging batch files and creating train/dev splits...")
    
    # Load and merge all batches
    all_batch_files = sorted(temp_manifest_dir.glob("batch_*.jsonl.gz"))
    
    # Calculate split sizes
    train_size = int(0.95 * total_processed)
    
    # Process batches and write directly to train/dev files
    train_cuts = []
    dev_cuts = []
    current_count = 0
    
    for batch_file in all_batch_files:
        batch_cuts = CutSet.from_file(batch_file)
        for cut in batch_cuts:
            if current_count < train_size:
                train_cuts.append(cut)
            else:
                dev_cuts.append(cut)
            current_count += 1
            
            # Save incrementally to avoid memory issues
            if len(train_cuts) >= batch_size:
                # Append to train file
                if not (output_dir / "elise_cuts_train.jsonl.gz").exists():
                    CutSet.from_cuts(train_cuts).to_file(output_dir / "elise_cuts_train.jsonl.gz")
                else:
                    # Append mode
                    with CutSet.open_writer(output_dir / "elise_cuts_train.jsonl.gz", mode="a") as writer:
                        for cut in train_cuts:
                            writer.write(cut)
                train_cuts = []
                
            if len(dev_cuts) >= batch_size:
                # Append to dev file
                if not (output_dir / "elise_cuts_dev.jsonl.gz").exists():
                    CutSet.from_cuts(dev_cuts).to_file(output_dir / "elise_cuts_dev.jsonl.gz")
                else:
                    # Append mode
                    with CutSet.open_writer(output_dir / "elise_cuts_dev.jsonl.gz", mode="a") as writer:
                        for cut in dev_cuts:
                            writer.write(cut)
                dev_cuts = []
    
    # Save any remaining cuts
    if train_cuts:
        if not (output_dir / "elise_cuts_train.jsonl.gz").exists():
            CutSet.from_cuts(train_cuts).to_file(output_dir / "elise_cuts_train.jsonl.gz")
        else:
            with CutSet.open_writer(output_dir / "elise_cuts_train.jsonl.gz", mode="a") as writer:
                for cut in train_cuts:
                    writer.write(cut)
    
    if dev_cuts:
        if not (output_dir / "elise_cuts_dev.jsonl.gz").exists():
            CutSet.from_cuts(dev_cuts).to_file(output_dir / "elise_cuts_dev.jsonl.gz")
        else:
            with CutSet.open_writer(output_dir / "elise_cuts_dev.jsonl.gz", mode="a") as writer:
                for cut in dev_cuts:
                    writer.write(cut)
    
    # Clean up temporary files
    logging.info("Cleaning up temporary files...")
    import shutil
    shutil.rmtree(temp_manifest_dir)
    
    logging.info(f"Saved {train_size} training cuts and {total_processed - train_size} dev cuts")
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
