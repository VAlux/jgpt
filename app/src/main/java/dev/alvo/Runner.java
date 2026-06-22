package dev.alvo;

import dev.alvo.grad.AutoGrad;
import dev.alvo.grad.AutoGradNode;
import dev.alvo.model.KVCache;
import dev.alvo.model.Model;
import dev.alvo.model.ModelParams;
import dev.alvo.model.Vector;
import dev.alvo.optimizer.AdamOptimizer;
import dev.alvo.repository.DataRepository;
import dev.alvo.repository.serialization.ModelDataSerializer.ModelData;
import dev.alvo.tokenizer.Tokenizer;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;

import static dev.alvo.grad.operation.GradNodeOperationDSL.div;
import static dev.alvo.grad.operation.GradNodeOperationDSL.neg;
import static dev.alvo.grad.operation.GradNodeOperationDSL.sum;

public class Runner {

  private static final int LOSS_MOVING_AVERAGE_WINDOW = 100;

  private final Model model;
  private final AutoGrad autoGrad;
  private final RandomGenerator randomGenerator;
  private final DataRepository dataRepository;

  public Runner(Model model, RandomGenerator randomGenerator, DataRepository dataRepository) {
    this.model = model;
    this.randomGenerator = randomGenerator;
    this.dataRepository = dataRepository;
    this.autoGrad = new AutoGrad();

//    this.byteModelRepository =
  }

  public ModelData train(Tokenizer tokenizer,
                         List<String> documents,
                         TrainingConfig config,
                         Random random) {

    var batchSize = config.batchSize();
    var iterationCount = config.iterationCount();
    var checkpointFrequency = config.checkpointFrequency();

    System.out.println("> Started tokenizer training...");
    tokenizer.train(documents);
    System.out.println("< Finished tokenizer training!");

    var params = ModelParams.create(
      tokenizer.getVocabularySize(),
      config.embeddingDimension(),
      config.maxSequenceLength(),
      config.transformerBlockCount(),
      4,
      config.attentionHeadCount(),
      random
    );

    var optimizer = new AdamOptimizer(config.learningRate(), params.flatten(), 0.9d, 0.999d, 1e-8d);

    var encodedDocs = documents.stream()
      .map(tokenizer::encode)
      .map(encoded -> Arrays.stream(encoded).limit(params.getMaxSequenceLength()).boxed().toList())
      .collect(Collectors.toCollection(() -> new ArrayList<>(documents.size())));

    Collections.shuffle(encodedDocs, random);

    var recentLosses = new ArrayDeque<Double>(LOSS_MOVING_AVERAGE_WINDOW);
    var recentLossSum = 0d;

    var batches = partition(encodedDocs, batchSize);
    for (int iteration = 0; iteration < iterationCount; iteration++) {
      var iterationStartTime = System.currentTimeMillis();

      var batch = batches.get(iteration % batches.size());

      var batchLosses = new ArrayList<AutoGradNode>();
      for (var document : batch) {
        var kvCache = new KVCache(params.transformerBlockCount());
        for (int positionId = 0; positionId < document.size() - 1; positionId++) {
          var tokenId = document.get(positionId);

          var outputLogits = model.gpt(
            tokenId,
            positionId,
            params,
            kvCache
          );

          var outputLogProbs = outputLogits.logSoftmax();
          var groundTruthTokenId = document.get(positionId + 1);
          var tokenLoss = neg(outputLogProbs.get(groundTruthTokenId));

          batchLosses.add(tokenLoss);
        }
      }

      var batchLoss = div(sum(batchLosses), batchLosses.size());
      this.autoGrad.backpropagate(batchLoss);

      if (recentLosses.size() == LOSS_MOVING_AVERAGE_WINDOW) {
        recentLossSum -= recentLosses.removeFirst();
      }

      recentLosses.addLast(batchLoss.value());
      recentLossSum += batchLoss.value();

      var movingAverageLoss = recentLossSum / recentLosses.size();

      optimizer.step();

      var finishedIterationIndex = iteration + 1;
      var iterationDuration = System.currentTimeMillis() - iterationStartTime;

      System.out.printf("Iteration %d/%d: loss = %.4f, moving average loss = %.4f, time = %.2fs\n%n",
        finishedIterationIndex,
        iterationCount,
        batchLoss.value(),
        movingAverageLoss,
        iterationDuration / 1000d);


      if (finishedIterationIndex % checkpointFrequency == 0) {
        System.out.printf("Checkpoint after %d", finishedIterationIndex);
        this.dataRepository.save(new ModelData(params, tokenizer), "iter-" + finishedIterationIndex);
        this.sample("", params, tokenizer, config.sampleCount(), config.temperature(), random);
      }
    }

    return new ModelData(params, tokenizer);
  }

  public static <T> List<List<T>> partition(List<T> list, int size) {
    var result = new ArrayList<List<T>>();

    for (int i = 0; i < list.size(); i += size) {
      result.add(list.subList(i, Math.min(i + size, list.size())));
    }

    return result;
  }

  public String predict(ModelParams params, Tokenizer tokenizer, String prompt, double temperature) {
    return predict(params, tokenizer, prompt, temperature, this.randomGenerator);
  }

  public String predict(ModelParams params, Tokenizer tokenizer, String prompt, double temperature, RandomGenerator rng) {
    final var encoded = tokenizer.encode(prompt);
    var tokens = new ArrayList<>(Arrays.stream(encoded).boxed().limit(encoded.length - 1).toList());

    if (tokens.size() > params.getMaxSequenceLength()) {
      throw new IllegalArgumentException("Prompt is too long");
    }

    var kvCache = new KVCache(params.transformerBlockCount());


    int positionId = 0;
    while (true) {
      // prefill
      var outputLogits = model.gpt(
        tokens.get(positionId),
        positionId,
        params,
        kvCache
      );

      positionId++;
      if (positionId < tokens.size()) {
        continue;
      }

      // decode
      int nextTokenId;
      if (temperature == 0) {
        nextTokenId = getGreedyTokenId(outputLogits);
      } else {
        var outputProbabilities = new Vector(outputLogits.nodes().stream().map(node -> div(node, temperature)).toList())
          .softmax()
          .nodes()
          .stream()
          .mapToDouble(AutoGradNode::value)
          .toArray();

        nextTokenId = getNextTokenId(outputProbabilities, rng);
      }

      tokens.add(nextTokenId);

      if (nextTokenId == tokenizer.eosToken() || tokens.size() > params.getMaxSequenceLength()) {
        break;
      }
    }

    return tokenizer.decode(tokens.stream().mapToInt(Integer::intValue).toArray());
  }

  private static int getGreedyTokenId(Vector logits) {
    var nodes = logits.nodes();

    int bestTokenId = 0;
    double bestLogit = nodes.getFirst().value();

    for (int i = 1; i < nodes.size(); i++) {
      double logit = nodes.get(i).value();
      if (logit > bestLogit) {
        bestLogit = logit;
        bestTokenId = i;
      }
    }

    return bestTokenId;
  }

  private static int getNextTokenId(double[] probabilities, RandomGenerator rng) {
    double total = 0;
    for (double probability : probabilities) {
      total += probability;
    }

    double randomThreshold = rng.nextDouble() * total;
    double acc = 0;

    for (int i = 0; i < probabilities.length; i++) {
      acc += probabilities[i];

      if (randomThreshold < acc) {
        return i;
      }
    }

    return probabilities.length - 1;
  }

  public String sample(String prompt,
                     ModelParams params,
                     Tokenizer tokenizer,
                     double temperature,
                     RandomGenerator rng) {

    return this.predict(params, tokenizer, prompt, temperature, rng);
  }

  public void sample(String prompt,
                     ModelParams params,
                     Tokenizer tokenizer,
                     int sampleCount,
                     double temperature,
                     RandomGenerator rng) {

    System.out.printf("> Started sampling with count: %d temperature: %f%n", sampleCount, temperature);

    for (int i = 0; i < sampleCount; i++) {
      var prediction = this.predict(params, tokenizer, prompt, temperature, rng);
      System.out.println(prediction);
    }

    System.out.println("< Finished sampling");
    System.out.println("--------------------");
  }
}
