package dev.alvo.model;

import dev.alvo.grad.AutoGradNode;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Stream;

public record ModelParams(
  Matrix tokenEmbeddings,
  Matrix positionEmbeddings,
  List<Matrix> attentionQuery,
  List<Matrix> attentionKey,
  List<Matrix> attentionValue,
  List<Matrix> attentionOutput,
  List<Matrix> mlpFullyConnected1,
  List<Matrix> mlpFullyConnected2,
  Matrix languageModelingHead,
  int attentionHeadCount,
  int attentionHeadDimension,
  int transformerBlockCount
) implements Iterable<AutoGradNode>, Serializable {

  private static final double STDDEV = 0.02;

  public static ModelParams create(int vocabSize,
                                   int embeddingDimension,
                                   int maxSequenceLength,
                                   int transformerBlockCount,
                                   int mlpFanoutFactor,
                                   int attentionHeadCount, Random random) {

    var tokenEmbeddings = Matrix.random(vocabSize, embeddingDimension, STDDEV, random);
    var positionEmbeddings = Matrix.random(maxSequenceLength, embeddingDimension, STDDEV, random);
    var hiddenDimension = mlpFanoutFactor * embeddingDimension;
    var attentionQuery = createMatrices(transformerBlockCount, embeddingDimension, embeddingDimension, random);
    var attentionKey = createMatrices(transformerBlockCount, embeddingDimension, embeddingDimension, random);
    var attentionValue = createMatrices(transformerBlockCount, embeddingDimension, embeddingDimension, random);
    var attentionOutput = createMatrices(transformerBlockCount, embeddingDimension, embeddingDimension, random);
    var mlpFullyConnected1 = createMatrices(transformerBlockCount, hiddenDimension, embeddingDimension, random);
    var mlpFullyConnected2 = createMatrices(transformerBlockCount, embeddingDimension, hiddenDimension, random);
    var languageModelingHead = Matrix.random(vocabSize, embeddingDimension, STDDEV, random);
    int attentionHeadDimension = embeddingDimension / attentionHeadCount;

    return new ModelParams(
      tokenEmbeddings,
      positionEmbeddings,
      attentionQuery,
      attentionKey,
      attentionValue,
      attentionOutput,
      mlpFullyConnected1,
      mlpFullyConnected2,
      languageModelingHead,
      attentionHeadCount,
      attentionHeadDimension,
      transformerBlockCount
    );
  }

  private static List<Matrix> createMatrices(int count, int rows, int cols, Random random) {
    var results = new ArrayList<Matrix>(count);

    for (int i = 0; i < count; i++) {
      results.add(Matrix.random(rows, cols, STDDEV, random));
    }

    return results;
  }

  public int getEmbeddingDimension() {
    if (tokenEmbeddings.vectors().isEmpty()) {
      return 0;
    }

    return tokenEmbeddings.get(0).nodes().size();
  }

  public int getVocabularySize() {
    return tokenEmbeddings.vectors().size();
  }

  public int getMaxSequenceLength() {
    return positionEmbeddings.vectors().size();
  }

  public List<AutoGradNode> flatten() {
    return Stream.of(
      this.tokenEmbeddings.nodes(),
      this.positionEmbeddings.nodes(),
      this.attentionQuery.stream().flatMap(Matrix::nodes),
      this.attentionKey.stream().flatMap(Matrix::nodes),
      this.attentionValue.stream().flatMap(Matrix::nodes),
      this.attentionOutput.stream().flatMap(Matrix::nodes),
      this.mlpFullyConnected1.stream().flatMap(Matrix::nodes),
      this.mlpFullyConnected2.stream().flatMap(Matrix::nodes),
      this.languageModelingHead.nodes()
    ).flatMap(Function.identity()).toList();
  }

  @Override
  public Iterator<AutoGradNode> iterator() {
    return this.flatten().iterator();
  }
}
