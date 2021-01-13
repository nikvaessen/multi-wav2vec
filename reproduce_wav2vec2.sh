set -e

# dependencies
#./install_fairseq.sh
#pip install -r requirements.txt

# download librispeech if not present
#daudio librispeech prepare --skip_extracting_if_exists

# prepare manifest
train_data_path="$(daudio librispeech path --dataset train 2> /dev/null)"
manifest_path="$PWD/manifest"
python wav2vec_fb/wav2vec_manifest.py \
  "$train_data_path" \
  --dest "$manifest_path" \
  --ext "flac" \
  --valid-percent 0.05 # change to 0 if overwriting

# overwrite val with librispeech dev-other
# don't for now

# train a wav2vec 2.0 base model
NUM_GPUS=1
UPDATE_FREQ=$(python -c "print(64//$NUM_GPUS)")
HYDRA_FULL_ERROR=1 fairseq-hydra-train \
  task.data="$manifest_path" \
  distributed_training.distributed_world_size=$NUM_GPUS \
  optimization.update_freq="[$UPDATE_FREQ]" \
  --config-dir wav2vec_fb/config/pretraining \
  --config-name wav2vec2_base_librispeech