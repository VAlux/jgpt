# jgpt

A tiny GPT, written from scratch in plain Java.

This is a small **educational project** — an excuse to build a working
transformer-based language model end to end without any machine learning
libraries. Everything (the autograd engine, the optimizer, the tokenizer and
the transformer itself) is implemented by hand so you can read it and follow
along.

It is inspired by Andrej Karpathy's
[microGPT](https://karpathy.github.io/2026/02/12/microgpt/).

## What it does

The model is a character-level language model. For example it can be trained on a list of Ukrainian
settlement names (`app/src/main/resources/ua-settlement-names.txt`) and then
generates new, made-up names in the same style. You can also give it a prefix
and let it complete the name.

Under the hood it includes a small handwritten autograd engine, an Adam
optimizer, a code-point tokenizer, and a single-block transformer with
multi-head attention — all kept intentionally tiny.

## Requirements

- Java 25 (the Gradle build will download a matching toolchain if needed)
- No other setup — the Gradle wrapper (`./gradlew`) is included

## How to run

The project ships **without a pre-trained model**, so the first run trains one
from scratch. Training is quick because the model is small, and it prints the
loss as it goes. When training finishes you drop into an interactive prompt.

```bash
./gradlew run --console=plain -q
```

(On Windows use `gradlew.bat run --console=plain -q`.)

The `--console=plain -q` flags keep Gradle's own progress bar and logging out of
the way, so the model's training output and the interactive prompt stay clean
and readable. They are recommended for the best experience.

During training the app periodically saves checkpoints into the `checkpoints/`
directory and prints a few sample names so you can watch it improve. Checkpoints
are written in the [safetensors](https://github.com/huggingface/safetensors)
format (`.safetensors`), so the weights can also be loaded by other tooling such
as PyTorch or NumPy.

Once training is done, type a prefix at the `|>` prompt and press Enter to get a
generated name, or just press Enter on an empty line (or type `/q`) to quit.

```
|> КИ
КИЇВКА
|> /q
Good bye!
```

### Reusing a trained model

After training, a model file is written to `checkpoints/`. To skip training and
load an existing model, pass its file name as an argument:

```bash
./gradlew run --console=plain -q --args="--model jgpt-vocab-38-seq-20-emb-16-trans-1-attn-4.safetensors"
```

## Command-line options

All options are passed to the app itself. With Gradle, wrap them in
`--args="..."` (e.g. `./gradlew run -q --args="--data names.txt --iteration-count 500"`);
with the native binary, pass them directly. Every option has a sensible default,
so the only one you normally need is `--data` (to train) or `--model` (to load).
Run with `--help` to see this list at any time.

### Input / output

| Option | Default | Description |
| --- | --- | --- |
| `--data <file>` | — | Training data file, one document per line. Required unless `--model` is given. |
| `--model <file>` | — | Load an existing model instead of training. The format is inferred from the file extension (`.safetensors` / `.json` / `.bin`). |
| `--type <format>` | `safetensors` | Save format for the trained model: `safetensors`, `json`, or `bin` (case-insensitive). |
| `--checkpoints-dir <dir>` | `checkpoints` | Directory where checkpoints and the final model are written. |
| `--checkpoint-frequency <n>` | `20` | Save a checkpoint every `n` training iterations. |
| `--sample-count <n>` | `10` | Number of sample names printed at each checkpoint. |

### Sampling

| Option | Default | Description |
| --- | --- | --- |
| `--temperature <t>` | `0.5` | Sampling temperature. Lower is more conservative/repetitive; higher is more random. |

### Training

| Option | Default | Description |
| --- | --- | --- |
| `--iteration-count <n>` | `1000` | Number of training iterations. |
| `--batch-size <n>` | `32` | Training batch size. |
| `--learning-rate <r>` | `0.01` | Adam optimizer learning rate. |
| `--seed <n>` | `1234` | RNG seed, for reproducible training. |

### Model architecture

These take effect only when training a new model (they are read from the file
when loading with `--model`).

| Option | Default | Description |
| --- | --- | --- |
| `--embedding-dim <n>` | `16` | Embedding dimension. |
| `--seq-length <n>` | `20` | Maximum sequence length. |
| `--transformer-blocks <n>` | `1` | Number of transformer blocks. |
| `--attention-heads <n>` | `4` | Number of attention heads. |

### Help

| Option | Description |
| --- | --- |
| `-h`, `--help` | Show usage help and exit. |
| `-V`, `--version` | Print version information and exit. |

## Building a native binary

The app can be compiled ahead-of-time into a single, self-contained native
executable with [GraalVM Native Image](https://www.graalvm.org/) — no JVM needed
to run it, and it starts instantly.

This needs a **GraalVM** JDK 25 on the build machine. The Gradle build selects a
GraalVM toolchain automatically; if you don't have one installed, point the build
at it with `JAVA_HOME` (e.g. `export JAVA_HOME=/path/to/graalvm-jdk-25`).

```bash
./gradlew nativeCompile
```

The binary is written to `app/build/native/nativeCompile/jgpt`. Run it like the
CLI above — train a fresh model, or load an existing one:

```bash
# Train from scratch, then drop into the interactive prompt
./app/build/native/nativeCompile/jgpt --data app/src/main/resources/ua-settlement-names.txt

# Or load a previously trained model and sample from it
./app/build/native/nativeCompile/jgpt --model checkpoints/jgpt-vocab-38-seq-20-emb-16-trans-1-attn-4.safetensors
```

All three model formats (`.safetensors`, `.json`, `.bin`) work in the native
binary. The picocli command metadata is generated at compile time by the
`picocli-codegen` annotation processor, and the Java-serialization metadata for
the `.bin` format lives in
`app/src/main/resources/META-INF/native-image/dev.alvo/jgpt/reachability-metadata.json`.

> Note: run the produced binary directly for the interactive prompt — Gradle's
> `nativeRun` task runs the executable with an empty stdin, so the REPL would
> exit immediately.

## Tests

```bash
./gradlew test
```
