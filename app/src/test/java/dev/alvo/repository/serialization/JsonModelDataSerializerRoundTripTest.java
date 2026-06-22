package dev.alvo.repository.serialization;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import dev.alvo.grad.AutoGradNode;
import dev.alvo.model.ModelParams;
import dev.alvo.repository.serialization.ModelDataSerializer.ModelData;
import dev.alvo.tokenizer.CodePointTokenizer;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JsonModelDataSerializerRoundTripTest {

  private static ModelData sampleModel() {
    var tokenizer = new CodePointTokenizer();
    tokenizer.train(List.of("kyiv", "lviv", "одеса"));
    var params = ModelParams.create(tokenizer.getVocabularySize(), 16, 20, 2, 4, 4, new Random(1));
    return new ModelData(params, tokenizer);
  }

  private static double[] weights(ModelParams params) {
    return params.flatten().stream().mapToDouble(AutoGradNode::value).toArray();
  }

  @Test
  void roundTripPreservesWeightsExactly() {
    var original = sampleModel();
    var serializer = new JsonModelDataSerializer();

    var loaded = serializer.deserialize(serializer.serialize(original)).orElseThrow();

    assertArrayEquals(weights(original.params()), weights(loaded.params()),
      "JSON must round-trip the double weights exactly");
  }

  @Test
  void roundTripPreservesArchitectureHyperParameters() {
    var original = sampleModel();
    var serializer = new JsonModelDataSerializer();

    var loaded = serializer.deserialize(serializer.serialize(original)).orElseThrow().params();
    var expected = original.params();

    assertEquals(expected.getVocabularySize(), loaded.getVocabularySize(), "vocabulary size");
    assertEquals(expected.getEmbeddingDimension(), loaded.getEmbeddingDimension(), "embedding dim");
    assertEquals(expected.getMaxSequenceLength(), loaded.getMaxSequenceLength(), "sequence length");
    assertEquals(expected.transformerBlockCount(), loaded.transformerBlockCount(), "block count");
    assertEquals(expected.attentionHeadCount(), loaded.attentionHeadCount(), "head count");
    assertEquals(expected.attentionHeadDimension(), loaded.attentionHeadDimension(), "head dim");
    assertEquals(expected.attentionQuery().size(), loaded.attentionQuery().size(), "per-block lists");
  }

  @Test
  void roundTripPreservesTokenizerBehaviour() {
    var original = sampleModel();
    var serializer = new JsonModelDataSerializer();

    var loaded = serializer.deserialize(serializer.serialize(original)).orElseThrow().tokenizer();

    assertEquals(original.tokenizer().getVocabulary(), loaded.getVocabulary(), "vocabulary");
    assertEquals(original.tokenizer().getVocabularySize(), loaded.getVocabularySize(), "vocab size");

    var text = "kyiv";
    assertArrayEquals(original.tokenizer().encode(text), loaded.encode(text), "encoding");
    assertEquals(text, loaded.decode(loaded.encode(text)), "decode(encode(text)) is stable");
  }

  @Test
  void producesReadableJsonWithRawWeights() {
    var serializer = new JsonModelDataSerializer();

    var json = new String(serializer.serialize(sampleModel()), StandardCharsets.UTF_8);

    // Pretty-printed output is multi-line and directly inspectable.
    assertTrue(json.contains("\n"), "output should be pretty-printed");

    var document = new Gson().fromJson(json, JsonObject.class);
    assertEquals("jgpt-json", document.get("format").getAsString());
    assertTrue(document.has("architecture"), "carries an architecture block");
    assertTrue(document.has("tokenizer"), "carries the tokenizer vocabulary");

    var params = document.getAsJsonObject("params");
    var tokenEmbeddings = params.getAsJsonArray("tokenEmbeddings");
    assertEquals(16, tokenEmbeddings.get(0).getAsJsonArray().size(),
      "each token embedding row exposes the raw embedding-dimension values");
    assertEquals(2, params.getAsJsonArray("blocks").size(), "one entry per transformer block");
  }

  @Test
  void deserializingGarbageReturnsEmpty() {
    var serializer = new JsonModelDataSerializer();
    assertTrue(serializer.deserialize("not json".getBytes(StandardCharsets.UTF_8)).isEmpty());
  }
}
