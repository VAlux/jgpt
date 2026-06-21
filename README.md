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

The model is a character-level language model. It trains on a list of Ukrainian
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
directory and prints a few sample names so you can watch it improve.

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
./gradlew run --console=plain -q --args="jgpt-vocab-38-seq-20-emb-16-trans-1-attn-4-iter-1000.bin"
```

## Tests

```bash
./gradlew test
```
