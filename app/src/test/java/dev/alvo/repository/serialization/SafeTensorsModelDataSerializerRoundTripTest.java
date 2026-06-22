package dev.alvo.repository.serialization;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import dev.alvo.grad.AutoGradNode;
import dev.alvo.model.ModelParams;
import dev.alvo.repository.serialization.ModelDataSerializer.ModelData;
import dev.alvo.tokenizer.CodePointTokenizer;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SafeTensorsModelDataSerializerRoundTripTest {

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
  void f64RoundTripPreservesWeightsExactly() {
    var original = sampleModel();
    var serializer = new SafeTensorsModelDataSerializer(SafeTensorsModelDataSerializer.Dtype.F64);

    var loaded = serializer.deserialize(serializer.serialize(original)).orElseThrow();

    assertArrayEquals(weights(original.params()), weights(loaded.params()),
      "F64 must round-trip the double weights bit-for-bit");
  }

  @Test
  void f32RoundTripPreservesWeightsWithinFloatPrecision() {
    var original = sampleModel();
    var serializer = new SafeTensorsModelDataSerializer(); // F32 by default

    var loaded = serializer.deserialize(serializer.serialize(original)).orElseThrow();

    assertArrayEquals(weights(original.params()), weights(loaded.params()), 1e-6,
      "F32 must round-trip the weights within float precision");
  }

  @Test
  void roundTripPreservesArchitectureHyperParameters() {
    var original = sampleModel();
    var serializer = new SafeTensorsModelDataSerializer();

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
    var serializer = new SafeTensorsModelDataSerializer();

    var loaded = serializer.deserialize(serializer.serialize(original)).orElseThrow().tokenizer();

    assertEquals(original.tokenizer().getVocabulary(), loaded.getVocabulary(), "vocabulary");
    assertEquals(original.tokenizer().getVocabularySize(), loaded.getVocabularySize(), "vocab size");

    var text = "kyiv";
    assertArrayEquals(original.tokenizer().encode(text), loaded.encode(text), "encoding");
    assertEquals(text, loaded.decode(loaded.encode(text)), "decode(encode(text)) is stable");
  }

  @Test
  void producesAWellFormedSafetensorsContainer() {
    var serializer = new SafeTensorsModelDataSerializer();

    var bytes = serializer.serialize(sampleModel());

    var buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    var headerLength = buffer.getLong();
    assertTrue(headerLength > 0 && headerLength <= bytes.length - Long.BYTES, "header length in range");

    var headerBytes = new byte[(int) headerLength];
    buffer.get(headerBytes);
    var header = new Gson().fromJson(new String(headerBytes, StandardCharsets.UTF_8), JsonObject.class);

    assertTrue(header.has("__metadata__"), "header carries a __metadata__ block");
    assertTrue(header.has("token_embeddings"), "header lists the token embedding tensor");
    assertTrue(header.has("lm_head"), "header lists the language modeling head tensor");
    assertEquals("F32", header.getAsJsonObject("token_embeddings").get("dtype").getAsString());

    // The last tensor's end offset must tile the buffer exactly: header + data == file size.
    var lmHeadEnd = header.getAsJsonObject("lm_head").getAsJsonArray("data_offsets").get(1).getAsLong();
    assertEquals(bytes.length, Long.BYTES + headerLength + lmHeadEnd, "offsets tile the data buffer");
  }

  @Test
  void deserializingGarbageReturnsEmpty() {
    var serializer = new SafeTensorsModelDataSerializer();
    assertTrue(serializer.deserialize(new byte[]{1, 2, 3, 4}).isEmpty());
  }
}
