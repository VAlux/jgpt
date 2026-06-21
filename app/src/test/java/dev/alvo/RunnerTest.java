package dev.alvo;

import dev.alvo.model.Matrix;
import dev.alvo.model.Model;
import dev.alvo.model.ModelParams;
import dev.alvo.repository.DataRepository;
import dev.alvo.repository.serialization.ModelDataSerializer.ModelData;
import dev.alvo.tokenizer.CodePointTokenizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.random.RandomGenerator;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RunnerTest {

  private static final double STDDEV = 0.02;
  private static final int EMBEDDING_DIMENSION = 8;
  private static final int HEAD_COUNT = 2;
  private static final int BLOCK_COUNT = 2;

  private CodePointTokenizer tokenizer;
  private RandomGenerator randomGenerator;
  private DataRepository dataRepository;

  @BeforeEach
  void setUp() {
    this.tokenizer = new CodePointTokenizer();
    this.tokenizer.train(List.of("hello brave new world"));
    this.randomGenerator = RandomGenerator.getDefault();
    this.dataRepository = mockDataRepository();
  }

  @Test
  void greedyDecodingIsDeterministic() {
    // temperature == 0 takes the argmax path, so repeated runs over identical
    // weights must produce identical output (no sampling involved).
    var params = buildParams(16, new Random(42));
    var runner = new Runner(new Model(), this.randomGenerator, this.dataRepository);

    String first = runner.predict(params, tokenizer, "hello", 0d);
    String second = runner.predict(params, tokenizer, "hello", 0d);

    assertNotNull(first);
    assertEquals(first, second);
  }

  @Test
  void samplingPathProducesOutputWithoutError() {
    // temperature > 0 exercises the softmax + cumulative-sampling branch.
    var params = buildParams(16, new Random(42));
    var runner = new Runner(new Model(), this.randomGenerator, this.dataRepository);

    String output = runner.predict(params, tokenizer, "hello", 1.0d);

    assertNotNull(output);
  }

  @Test
  void handlesEmptyPrompt() {
    // encode("") is [BOS, EOS]; predict drops the trailing EOS and still generates.
    var params = buildParams(16, new Random(42));
    var runner = new Runner(new Model(), this.randomGenerator, this.dataRepository);

    String output = runner.predict(params, tokenizer, "", 0d);

    assertNotNull(output);
  }

  @Test
  void throwsWhenPromptExceedsMaxSequenceLength() {
    var params = buildParams(2, new Random(42));
    var runner = new Runner(new Model(), this.randomGenerator, this.dataRepository);

    var exception = assertThrows(
      IllegalArgumentException.class,
      () -> runner.predict(params, tokenizer, "hello brave new world", 0d)
    );
    assertEquals("Prompt is too long", exception.getMessage());
  }

  private ModelParams buildParams(int maxSequenceLength, Random random) {
    int vocabSize = tokenizer.getVocabularySize();
    return new ModelParams(
      Matrix.random(vocabSize, EMBEDDING_DIMENSION, STDDEV, random),
      Matrix.random(maxSequenceLength, EMBEDDING_DIMENSION, STDDEV, random),
      squareMatrices(random),
      squareMatrices(random),
      squareMatrices(random),
      squareMatrices(random),
      squareMatrices(random),
      squareMatrices(random),
      Matrix.random(vocabSize, EMBEDDING_DIMENSION, STDDEV, random),
      HEAD_COUNT,
      EMBEDDING_DIMENSION / HEAD_COUNT,
      BLOCK_COUNT
    );
  }

  private static List<Matrix> squareMatrices(Random random) {
    var matrices = new ArrayList<Matrix>();
    for (int i = 0; i < BLOCK_COUNT; i++) {
      matrices.add(Matrix.random(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION, STDDEV, random));
    }
    return matrices;
  }

  private static DataRepository mockDataRepository() {
    return new DataRepository() {
      @Override
      public void save(ModelData data) {
      }

      @Override
      public void save(ModelData data, String suffix) {
      }

      @Override
      public Optional<ModelData> load(String fileName) {
        return Optional.empty();
      }

      @Override
      public boolean exists(String path) {
        return false;
      }
    };
  }
}
