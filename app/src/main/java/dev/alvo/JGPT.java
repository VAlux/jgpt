package dev.alvo;

import dev.alvo.model.Model;
import dev.alvo.repository.DataRepository;
import dev.alvo.repository.ModelDataRepository;
import dev.alvo.repository.ModelNameBuilder;
import dev.alvo.repository.serialization.ByteArrayModelDataSerializer;
import dev.alvo.repository.serialization.ModelDataSerializer.ModelData;
import dev.alvo.repository.storage.ByteArrayModelDataStorage;
import dev.alvo.tokenizer.CodePointTokenizer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Random;
import java.util.random.RandomGenerator;

public class JGPT {

  private static final String CHECKPOINTS_DIRECTORY = "checkpoints";

  static void main(String[] args) {
    trapExceptions();

    var dataset = loadDataset("ua-settlement-names.txt");

    var random = new Random(1234);
    var randomGenerator = RandomGenerator.getDefault();
    var nameBuilder = new ModelNameBuilder();

    // Simple java serialization for now.
    var repository = new ModelDataRepository<>(
      CHECKPOINTS_DIRECTORY,
      new ByteArrayModelDataSerializer(),
      new ByteArrayModelDataStorage(),
      nameBuilder);

    var model = new Model();
    var runner = new Runner(model, randomGenerator, repository);
    var tokenizer = new CodePointTokenizer();

    var trainedModelData = getModelFileName(args)
      .map(fileName -> repository.load(fileName).orElseThrow(() -> new IllegalStateException(
        "Error loading model '" + fileName + "'. Searched: " + repository.searchedLocations(fileName))))
      .orElseGet(() -> getTrainedModel(runner, tokenizer, dataset, repository, random));

    var consoleReader = new BufferedReader(new InputStreamReader(System.in));
    while (true) {
      System.out.print("|> ");

      var input = readLine(consoleReader);
      if (input == null || input.isBlank() || input.equals("/q")) {
        System.out.println("Good bye!");
        System.exit(0);
      }

      var result = runner.sample(
        input.toUpperCase(Locale.ROOT),
        trainedModelData.params(),
        trainedModelData.tokenizer(),
        0.5d,
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

  private static Optional<String> getModelFileName(String[] args) {
    if (args.length > 0) {
      return Optional.of(args[0]);
    }

    return Optional.empty();
  }

  private static ModelData getTrainedModel(Runner runner,
                                           CodePointTokenizer tokenizer,
                                           List<String> dataset,
                                           DataRepository repository,
                                           Random random) {

    System.out.println("Starting model training...");
    var trainedModelData = runner.train(tokenizer, dataset, 32, 1000, 20, random);
    repository.save(trainedModelData);

    return trainedModelData;
  }

  private static List<String> loadDataset(String datasetName) {
    try (var stream = JGPT.class.getClassLoader().getResourceAsStream(datasetName)) {
      if (stream == null) {
        throw new RuntimeException("Could not load dataset content");
      }

      return new String(stream.readAllBytes()).lines().map(String::trim).toList();
    } catch (IOException e) {
      throw new RuntimeException("Cannot open dataset file");
    }
  }

  private static void trapExceptions() {
    Thread.setDefaultUncaughtExceptionHandler((thread, throwable) ->
      System.err.println("Uncaught exception in thread " + thread.getName() + ": " + throwable.getMessage()));
  }
}
