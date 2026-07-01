#!/usr/bin/env bash
#
# Run the jgpt training loop via Gradle on the Ukrainian proverbs dataset.
#
# The model is tuned to invent new, absurd-but-plausible Ukrainian sayings.
# Every option below mirrors a CLI flag of dev.alvo.cli.JGPTCommand. Edit the
# values you care about and run:
#
#   ./train_proverbs.sh
#
# Anything not listed here falls back to the application's own defaults.

set -euo pipefail

# Resolve the project root so the script works from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- JVM heap ----------------------------------------------------------------
#
# The proverbs model (deeper/wider, seq 100) builds a large autograd graph, so
# the JVM's default max heap overflows with "Java heap space". JAVA_TOOL_OPTIONS
# is picked up automatically by the JVM that Gradle forks to run the app.
# Override the size per-run with e.g.  JVM_HEAP=12g ./train_proverbs.sh
JVM_HEAP="${JVM_HEAP:-16g}"
export JAVA_TOOL_OPTIONS="-Xmx${JVM_HEAP} ${JAVA_TOOL_OPTIONS:-}"

# --- Training parameters -----------------------------------------------------

# Save format for the trained model: safetensors | json | bin
TYPE="safetensors"

# Training data file, one document per line. Required to train a new model.
# ua-proverbs.txt = full Ukrainian sayings -> the model invents new "proverbs".
DATA="data/ua-proverbs.txt"

# Load an existing model instead of training (format inferred from extension).
# Leave empty to train from scratch.
MODEL=""

# Sampling temperature. Higher = more creative / absurd proverbs (that's the fun);
# lower = safer but more likely to echo the training set. 0.8 is a good sweet spot.
TEMPERATURE="0.8"

# Training batch size. 32 roughly halves per-iteration cost vs 64 with no loss of
# model capacity (just noisier gradients, which tends to help creative output).
BATCH_SIZE="32"

# Number of training iterations. A 963-line dataset converges/overfits well before
# 6000, so 3500 saves wall-clock; watch the checkpoint samples and stop early.
ITERATION_COUNT="500"

# Save a checkpoint every N iterations. Spaced out so you can pick an earlier
# checkpoint that recombines fragments before it overfits into verbatim recall.
CHECKPOINT_FREQUENCY="10"

# RNG seed.
SEED="1234"

# Checkpoint output directory.
CHECKPOINTS_DIR="checkpoints"

# Samples printed at each checkpoint. More = more chances at a funny one.
SAMPLE_COUNT="15"

# Model embedding dimension. Proverbs need more capacity than the 16 used for names.
EMBEDDING_DIM="64"

# Maximum sequence length. Dominates per-iteration cost. 64 fully covers ~94% of
# proverbs (mean 40, p95 65 chars); only long tails get truncated. 100 was wasteful.
SEQ_LENGTH="64"

# Number of transformer blocks. Depth helps model sentence-level structure.
TRANSFORMER_BLOCKS="3"

# Number of attention heads. Must divide EMBEDDING_DIM (64 / 8 = 8 dims/head).
ATTENTION_HEADS="8"

# Adam learning rate. Lower than the name-model default for stability at depth 3.
LEARNING_RATE="0.005"

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
