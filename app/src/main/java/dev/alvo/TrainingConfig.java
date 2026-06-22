package dev.alvo;

/**
 * Bundles the training and model-architecture hyperparameters supplied on the command line and
 * passed to {@link Runner#train}. Grouping them in one carrier keeps the {@code train} signature
 * readable as the set of tunables grows.
 */
public record TrainingConfig(
  int batchSize,
  int iterationCount,
  int checkpointFrequency,
  int sampleCount,
  int embeddingDimension,
  int maxSequenceLength,
  int transformerBlockCount,
  int attentionHeadCount,
  double learningRate,
  double temperature) {
}
