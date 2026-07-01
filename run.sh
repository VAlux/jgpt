#!/usr/bin/env bash
#
# Run an already-trained jgpt model via Gradle.
#
# This skips training entirely: it loads a previously saved model and drops
# straight into the interactive sampling REPL. The options below mirror the
# CLI flags of dev.alvo.cli.JGPTCommand that are relevant when loading a model,
# pre-filled with their default values. Point MODEL at your saved model and run:
#
#   ./run.sh
#
# (The model's format is inferred from its file extension: .safetensors / .json / .bin)

set -euo pipefail

# Resolve the project root so the script works from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Run parameters (defaults match JGPTCommand) -----------------------------

# Existing model to load; format is inferred from the file extension. Required.
MODEL="checkpoints/jgpt-vocab-43-seq-30-emb-16-trans-1-attn-4.safetensors"

# Directory the loader searches for the model.
CHECKPOINTS_DIR="checkpoints"

# Sampling temperature used by the REPL.
TEMPERATURE="1.0"

# --- Assemble CLI arguments --------------------------------------------------

ARGS=(
  --model "$MODEL"
  --checkpoints-dir "$CHECKPOINTS_DIR"
  --temperature "$TEMPERATURE"
)

# --- Run via Gradle ----------------------------------------------------------

# Quote each argument so it survives Gradle's --args parsing.
GRADLE_ARGS=""
for arg in "${ARGS[@]}"; do
  GRADLE_ARGS+="\"$arg\" "
done

echo "Running: ./gradlew run --args='${GRADLE_ARGS}'"
exec ./gradlew run --console=plain -q --args="$GRADLE_ARGS"
