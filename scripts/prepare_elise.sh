#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=0
stop_stage=3
sampling_rate=24000  # Default, will be overridden by actual audio sampling rate
nj=20

. scripts/parse_options.sh || exit 1

mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Prepare Elise manifests"
  python3 prepare_elise.py
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Extract Fbank features"
  mkdir -p data/fbank
  
  # The existing compute_fbank.py will handle the manifests correctly
  for subset in train dev; do
    python3 tools/compute_fbank.py \
      --source-dir data/manifests \
      --dest-dir data/fbank \
      --dataset elise \
      --subset ${subset} \
      --sampling-rate 24000 \
      --frame-shift 256 \
      --frame-length 1024 \
      --num-mel-bins 100 \
      --num-jobs ${nj}
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Generate token file"
  python3 tools/prepare_token_file_elise.py
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Validate manifests"
  python3 tools/validate_manifest.py data/fbank/elise_cuts_train.jsonl.gz
fi

echo "Data preparation completed!"
