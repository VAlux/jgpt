package dev.alvo.cli;

import dev.alvo.Inference;
import dev.alvo.TrainingConfig;
import dev.alvo.model.Model;
import dev.alvo.repository.DataRepository;
import dev.alvo.repository.ModelDataRepository;
import dev.alvo.repository.ModelNameBuilder;
import dev.alvo.repository.serialization.ModelDataSerializer.ModelData;
import dev.alvo.repository.storage.ByteArrayModelDataStorage;
import dev.alvo.tokenizer.CodePointTokenizer;

import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Random;
import java.util.random.RandomGenerator;

/**
 * The {@code jgpt} command. Trains a model on a text dataset (or loads a previously trained one) and
 * then drops into an interactive sampling REPL.
 */
@Command(
  name = "jgpt",
  mixinStandardHelpOptions = true,
  version = "jgpt 1.0",
  description = "Train a tiny GPT on a text dataset and sample from it interactively.")
public class JGPTCommand implements Runnable {

  @Option(names = "--type", defaultValue = "safetensors",
    description = "Save format for the trained model: ${COMPLETION-CANDIDATES} (default: ${DEFAULT-VALUE}).")
  private ModelFormat type;

  @Option(names = "--model",
    description = "Load an existing model instead of training; format is inferred from the file extension "
      + "(.safetensors/.json/.bin).")
  private Path model;

  @Option(names = "--data",
    description = "Training data file, one document per line. Required unless --model is given.")
  private Path data;

  @Option(names = "--temperature", defaultValue = "0.5",
    description = "Sampling temperature (default: ${DEFAULT-VALUE}).")
  private double temperature;

  @Option(names = "--batch-size", defaultValue = "32",
    description = "Training batch size (default: ${DEFAULT-VALUE}).")
  private int batchSize;

  @Option(names = "--iteration-count", defaultValue = "1000",
    description = "Number of training iterations (default: ${DEFAULT-VALUE}).")
  private int iterationCount;

  @Option(names = "--checkpoint-frequency", defaultValue = "20",
    description = "Save a checkpoint every N iterations (default: ${DEFAULT-VALUE}).")
  private int checkpointFrequency;

  @Option(names = "--seed", defaultValue = "1234",
    description = "RNG seed (default: ${DEFAULT-VALUE}).")
  private long seed;

  @Option(names = "--checkpoints-dir", defaultValue = "checkpoints",
    description = "Checkpoint output directory (default: ${DEFAULT-VALUE}).")
  private Path checkpointsDir;

  @Option(names = "--sample-count", defaultValue = "10",
    description = "Samples printed at each checkpoint (default: ${DEFAULT-VALUE}).")
  private int sampleCount;

  @Option(names = "--embedding-dim", defaultValue = "16",
    description = "Model embedding dimension (default: ${DEFAULT-VALUE}).")
  private int embeddingDim;

  @Option(names = "--seq-length", defaultValue = "20",
    description = "Maximum sequence length (default: ${DEFAULT-VALUE}).")
  private int seqLength;

  @Option(names = "--transformer-blocks", defaultValue = "1",
    description = "Number of transformer blocks (default: ${DEFAULT-VALUE}).")
  private int transformerBlocks;

  @Option(names = "--attention-heads", defaultValue = "4",
    description = "Number of attention heads (default: ${DEFAULT-VALUE}).")
  private int attentionHeads;

  @Option(names = "--learning-rate", defaultValue = "0.01",
    description = "Adam learning rate (default: ${DEFAULT-VALUE}).")
  private double learningRate;

  @Override
  public void run() {
    var random = new Random(seed);
    var randomGenerator = RandomGenerator.getDefault();

    var gptModel = new Model();
    var tokenizer = new CodePointTokenizer();

    // Checkpoints and the final model are written in the format chosen via --type.
    var saveRepository = new ModelDataRepository<>(
      checkpointsDir.toString(),
      type.serializer(),
      new ByteArrayModelDataStorage(),
      new ModelNameBuilder(type.extension()));

    var runner = new Inference(gptModel, randomGenerator, saveRepository);

    var trainedModelData = loadExistingModel()
      .orElseGet(() -> trainModel(runner, tokenizer, saveRepository, random));

    repl(runner, trainedModelData, randomGenerator);
  }

  private Optional<ModelData> loadExistingModel() {
    if (this.model == null) {
      return Optional.empty();
    }

    var fileName = this.model.toString();
    var format = ModelFormat.fromExtension(fileName);

    // Loading does not consult the name builder's extension, but the constructor requires one;
    // use the inferred format's extension for consistency.
    var loadRepository = new ModelDataRepository<>(
      checkpointsDir.toString(),
      format.serializer(),
      new ByteArrayModelDataStorage(),
      new ModelNameBuilder(format.extension()));

    var loaded = loadRepository.load(fileName)
      .orElseThrow(() -> new IllegalStateException(
        "Error loading model '" + fileName + "'. Searched: " + loadRepository.searchedLocations(fileName)));

    return Optional.of(loaded);
  }

  private ModelData trainModel(Inference inference,
                               CodePointTokenizer tokenizer,
                               DataRepository repository,
                               Random random) {

    var dataset = loadDataset();
    var config = new TrainingConfig(
      batchSize,
      iterationCount,
      checkpointFrequency,
      sampleCount,
      embeddingDim,
      seqLength,
      transformerBlocks,
      attentionHeads,
      learningRate,
      temperature);

    System.out.println("Starting model training...");
    var trainedModelData = inference.train(tokenizer, dataset, config, random);
    repository.save(trainedModelData);

    return trainedModelData;
  }

  private List<String> loadDataset() {
    if (this.data == null) {
      throw new IllegalArgumentException("Missing required option '--data' (needed to train a new model).");
    }

    try {
      return Files.readAllLines(this.data).stream().map(String::trim).toList();
    } catch (IOException e) {
      throw new UncheckedIOException("Cannot read training data: " + this.data, e);
    }
  }

  private void repl(Inference inference, ModelData trainedModelData, RandomGenerator randomGenerator) {
    var consoleReader = new BufferedReader(new InputStreamReader(System.in));
    while (true) {
      System.out.print("|> ");

      var input = readLine(consoleReader);
      if (input == null || input.isBlank() || input.equals("/q")) {
        System.out.println("Good bye!");
        return;
      }

      var result = inference.sample(
        input.toUpperCase(Locale.ROOT),
        trainedModelData.params(),
        trainedModelData.tokenizer(),
        temperature,
        randomGenerator);

      System.out.println(result);
    }
  }

  private static String readLine(BufferedReader reader) {
    try {
      return reader.readLine();
    } catch (IOException e) {
      throw new UncheckedIOException("Failed to read input", e);
    }
  }
}
