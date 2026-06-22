package dev.alvo.repository.serialization;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import dev.alvo.grad.AutoGradNode;
import dev.alvo.model.Matrix;
import dev.alvo.model.ModelParams;
import dev.alvo.model.Vector;
import dev.alvo.tokenizer.CodePointTokenizer;
import dev.alvo.tokenizer.Tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Optional;

/**
 * Serializes a {@link ModelData} to a plain, human-readable JSON document.
 *
 * <p>Unlike {@link SafeTensorsModelDataSerializer}, which packs the weights into an opaque binary
 * buffer, this serializer writes every weight out as nested JSON arrays of numbers. The result is
 * not compact, but it gives a direct, inspectable view of the raw model: the architecture
 * hyper-parameters, the tokenizer vocabulary, and each weight matrix laid out row by row. Because
 * the values travel through JSON as ordinary numbers, the {@code double} weights round-trip exactly.
 *
 * <p>The document shape is:
 * <pre>{@code
 * {
 *   "format": "jgpt-json",
 *   "architecture": { "vocabularySize": .., "embeddingDimension": .., "maxSequenceLength": ..,
 *                     "attentionHeadCount": .., "attentionHeadDimension": .., "transformerBlockCount": .. },
 *   "tokenizer": { "class": "..", "vocabulary": [ <code points> ] },
 *   "params": {
 *     "tokenEmbeddings": [[..],[..]],
 *     "positionEmbeddings": [[..]],
 *     "blocks": [ { "attentionQuery": [[..]], "attentionKey": [[..]], "attentionValue": [[..]],
 *                  "attentionOutput": [[..]], "mlpFullyConnected1": [[..]], "mlpFullyConnected2": [[..]] } ],
 *     "languageModelingHead": [[..]]
 *   }
 * }
 * }</pre>
 */
public final class JsonModelDataSerializer implements ModelDataSerializer<byte[]> {

  private static final String FORMAT = "format";
  private static final String FORMAT_VALUE = "jgpt-json";
  private static final String ARCHITECTURE = "architecture";
  private static final String VOCABULARY_SIZE = "vocabularySize";
  private static final String EMBEDDING_DIMENSION = "embeddingDimension";
  private static final String MAX_SEQUENCE_LENGTH = "maxSequenceLength";
  private static final String ATTENTION_HEAD_COUNT = "attentionHeadCount";
  private static final String ATTENTION_HEAD_DIMENSION = "attentionHeadDimension";
  private static final String TRANSFORMER_BLOCK_COUNT = "transformerBlockCount";
  private static final String TOKENIZER = "tokenizer";
  private static final String TOKENIZER_CLASS = "class";
  private static final String VOCABULARY = "vocabulary";
  private static final String PARAMS = "params";
  private static final String TOKEN_EMBEDDINGS = "tokenEmbeddings";
  private static final String POSITION_EMBEDDINGS = "positionEmbeddings";
  private static final String BLOCKS = "blocks";
  private static final String ATTENTION_QUERY = "attentionQuery";
  private static final String ATTENTION_KEY = "attentionKey";
  private static final String ATTENTION_VALUE = "attentionValue";
  private static final String ATTENTION_OUTPUT = "attentionOutput";
  private static final String MLP_FULLY_CONNECTED_1 = "mlpFullyConnected1";
  private static final String MLP_FULLY_CONNECTED_2 = "mlpFullyConnected2";
  private static final String LANGUAGE_MODELING_HEAD = "languageModelingHead";

  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  @Override
  public byte[] serialize(ModelData data) {
    var params = data.params();

    var document = new JsonObject();
    document.addProperty(FORMAT, FORMAT_VALUE);
    document.add(ARCHITECTURE, buildArchitecture(params));
    document.add(TOKENIZER, buildTokenizer(data.tokenizer()));
    document.add(PARAMS, buildParams(params));

    return GSON.toJson(document).getBytes(StandardCharsets.UTF_8);
  }

  @Override
  public Optional<ModelData> deserialize(byte[] bytes) {
    try {
      var document = GSON.fromJson(new String(bytes, StandardCharsets.UTF_8), JsonObject.class);
      var architecture = document.getAsJsonObject(ARCHITECTURE);
      var params = document.getAsJsonObject(PARAMS);

      var headCount = architecture.get(ATTENTION_HEAD_COUNT).getAsInt();
      var headDimension = architecture.get(ATTENTION_HEAD_DIMENSION).getAsInt();
      var blockCount = architecture.get(TRANSFORMER_BLOCK_COUNT).getAsInt();

      var blocks = params.getAsJsonArray(BLOCKS);
      var attentionQuery = new ArrayList<Matrix>(blockCount);
      var attentionKey = new ArrayList<Matrix>(blockCount);
      var attentionValue = new ArrayList<Matrix>(blockCount);
      var attentionOutput = new ArrayList<Matrix>(blockCount);
      var mlpFullyConnected1 = new ArrayList<Matrix>(blockCount);
      var mlpFullyConnected2 = new ArrayList<Matrix>(blockCount);
      for (var block = 0; block < blockCount; block++) {
        var blockObject = blocks.get(block).getAsJsonObject();
        attentionQuery.add(readMatrix(blockObject, ATTENTION_QUERY));
        attentionKey.add(readMatrix(blockObject, ATTENTION_KEY));
        attentionValue.add(readMatrix(blockObject, ATTENTION_VALUE));
        attentionOutput.add(readMatrix(blockObject, ATTENTION_OUTPUT));
        mlpFullyConnected1.add(readMatrix(blockObject, MLP_FULLY_CONNECTED_1));
        mlpFullyConnected2.add(readMatrix(blockObject, MLP_FULLY_CONNECTED_2));
      }

      var modelParams = new ModelParams(
        readMatrix(params, TOKEN_EMBEDDINGS),
        readMatrix(params, POSITION_EMBEDDINGS),
        attentionQuery,
        attentionKey,
        attentionValue,
        attentionOutput,
        mlpFullyConnected1,
        mlpFullyConnected2,
        readMatrix(params, LANGUAGE_MODELING_HEAD),
        headCount,
        headDimension,
        blockCount);

      return Optional.of(new ModelData(modelParams, reconstructTokenizer(document.getAsJsonObject(TOKENIZER))));
    } catch (RuntimeException e) {
      System.err.println("Error deserializing JSON model: " + e.getMessage());
      return Optional.empty();
    }
  }

  private static JsonObject buildArchitecture(ModelParams params) {
    var architecture = new JsonObject();
    architecture.addProperty(VOCABULARY_SIZE, params.getVocabularySize());
    architecture.addProperty(EMBEDDING_DIMENSION, params.getEmbeddingDimension());
    architecture.addProperty(MAX_SEQUENCE_LENGTH, params.getMaxSequenceLength());
    architecture.addProperty(ATTENTION_HEAD_COUNT, params.attentionHeadCount());
    architecture.addProperty(ATTENTION_HEAD_DIMENSION, params.attentionHeadDimension());
    architecture.addProperty(TRANSFORMER_BLOCK_COUNT, params.transformerBlockCount());
    return architecture;
  }

  private static JsonObject buildTokenizer(Tokenizer tokenizer) {
    var vocabulary = new JsonArray();
    tokenizer.getVocabulary().forEach(vocabulary::add);

    var tokenizerObject = new JsonObject();
    tokenizerObject.addProperty(TOKENIZER_CLASS, tokenizer.getClass().getName());
    tokenizerObject.add(VOCABULARY, vocabulary);
    return tokenizerObject;
  }

  private static JsonObject buildParams(ModelParams params) {
    var paramsObject = new JsonObject();
    paramsObject.add(TOKEN_EMBEDDINGS, matrixToJson(params.tokenEmbeddings()));
    paramsObject.add(POSITION_EMBEDDINGS, matrixToJson(params.positionEmbeddings()));

    var blocks = new JsonArray();
    for (var block = 0; block < params.transformerBlockCount(); block++) {
      var blockObject = new JsonObject();
      blockObject.add(ATTENTION_QUERY, matrixToJson(params.attentionQuery().get(block)));
      blockObject.add(ATTENTION_KEY, matrixToJson(params.attentionKey().get(block)));
      blockObject.add(ATTENTION_VALUE, matrixToJson(params.attentionValue().get(block)));
      blockObject.add(ATTENTION_OUTPUT, matrixToJson(params.attentionOutput().get(block)));
      blockObject.add(MLP_FULLY_CONNECTED_1, matrixToJson(params.mlpFullyConnected1().get(block)));
      blockObject.add(MLP_FULLY_CONNECTED_2, matrixToJson(params.mlpFullyConnected2().get(block)));
      blocks.add(blockObject);
    }
    paramsObject.add(BLOCKS, blocks);

    paramsObject.add(LANGUAGE_MODELING_HEAD, matrixToJson(params.languageModelingHead()));
    return paramsObject;
  }

  private static JsonArray matrixToJson(Matrix matrix) {
    var rows = new JsonArray();
    for (var vector : matrix.vectors()) {
      var row = new JsonArray();
      for (var node : vector.nodes()) {
        row.add(node.value());
      }
      rows.add(row);
    }
    return rows;
  }

  private static Matrix readMatrix(JsonObject owner, String name) {
    var rows = owner.getAsJsonArray(name);
    if (rows == null) {
      throw new IllegalArgumentException("Missing weight matrix in JSON model: " + name);
    }

    var vectors = new ArrayList<Vector>(rows.size());
    for (var row : rows) {
      var values = row.getAsJsonArray();
      var nodes = new ArrayList<AutoGradNode>(values.size());
      for (var value : values) {
        nodes.add(new AutoGradNode(value.getAsDouble()));
      }
      vectors.add(new Vector(nodes));
    }

    return new Matrix(vectors);
  }

  private static Tokenizer reconstructTokenizer(JsonObject tokenizerObject) {
    var tokenizer = new CodePointTokenizer();
    var vocabulary = tokenizerObject.getAsJsonArray(VOCABULARY);

    if (vocabulary != null && !vocabulary.isEmpty()) {
      // CodePointTokenizer derives its vocabulary from the distinct, sorted code points of the
      // training documents. Feeding each stored code point back as a one-character document
      // therefore reproduces the original vocabulary exactly.
      var documents = new ArrayList<String>(vocabulary.size());
      for (var codePoint : vocabulary) {
        documents.add(new String(Character.toChars(codePoint.getAsInt())));
      }
      tokenizer.train(documents);
    }

    return tokenizer;
  }
}
