#!/usr/bin/env bash
#
# Run the jgpt training loop via Gradle on the Ukrainian settlements dataset.
#
# The model learns character patterns of place names and invents new,
# plausible-sounding Ukrainian settlement names.
# Every option below mirrors a CLI flag of dev.alvo.cli.JGPTCommand and is
# pre-filled with that flag's default value. Edit the values you care about
# (at minimum point DATA at your dataset) and run:
#
#   ./train_settlements.sh
#
# Anything not listed here falls back to the application's own defaults.

set -euo pipefail

# Resolve the project root so the script works from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- JVM heap ----------------------------------------------------------------
#
# Give the forked training JVM a larger max heap; the handwritten autograd graph
# can be memory-hungry. JAVA_TOOL_OPTIONS is picked up automatically by the JVM
# that Gradle forks to run the app. Override per-run with e.g.
#   JVM_HEAP=8g ./train_settlements.sh
JVM_HEAP="${JVM_HEAP:-4g}"
export JAVA_TOOL_OPTIONS="-Xmx${JVM_HEAP} ${JAVA_TOOL_OPTIONS:-}"

# --- Training parameters (defaults match JGPTCommand) ------------------------

# Save format for the trained model: safetensors | json | bin
TYPE="safetensors"

# Training data file, one document per line. Required to train a new model.
DATA="data/ua-settlements-geonames.txt"

# Load an existing model instead of training (format inferred from extension).
# Leave empty to train from scratch.
MODEL=""

# Sampling temperature.
TEMPERATURE="0.5"

# Training batch size.
BATCH_SIZE="32"

# Number of training iterations.
ITERATION_COUNT="1000"

# Save a checkpoint every N iterations.
CHECKPOINT_FREQUENCY="20"

# RNG seed.
SEED="1234"

# Checkpoint output directory.
CHECKPOINTS_DIR="checkpoints"

# Samples printed at each checkpoint.
SAMPLE_COUNT="10"

# Model embedding dimension.
EMBEDDING_DIM="16"

# Maximum sequence length.
SEQ_LENGTH="30"

# Number of transformer blocks.
TRANSFORMER_BLOCKS="1"

# Number of attention heads.
ATTENTION_HEADS="4"

# Adam learning rate.
LEARNING_RATE="0.01"

# --- Assemble CLI arguments --------------------------------------------------

ARGS=(
  --type "$TYPE"
  --temperature "$TEMPERATURE"
  --batch-size "$BATCH_SIZE"
  --iteration-count "$ITERATION_COUNT"
  --checkpoint-frequency "$CHECKPOINT_FREQUENCY"
  --seed "$SEED"
  --checkpoints-dir "$CHECKPOINTS_DIR"
  --sample-count "$SAMPLE_COUNT"
  --embedding-dim "$EMBEDDING_DIM"
  --seq-length "$SEQ_LENGTH"
  --transformer-blocks "$TRANSFORMER_BLOCKS"
  --attention-heads "$ATTENTION_HEADS"
  --learning-rate "$LEARNING_RATE"
)

# --data is required when training from scratch; --model takes precedence when set.
if [[ -n "$MODEL" ]]; then
  ARGS+=(--model "$MODEL")
else
  ARGS+=(--data "$DATA")
fi

# --- Run via Gradle ----------------------------------------------------------

# Quote each argument so it survives Gradle's --args parsing.
GRADLE_ARGS=""
for arg in "${ARGS[@]}"; do
  GRADLE_ARGS+="\"$arg\" "
done

echo "Running: ./gradlew run --args='${GRADLE_ARGS}'"
exec ./gradlew run --console=plain -q --args="$GRADLE_ARGS"
