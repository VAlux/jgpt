package dev.alvo.repository.serialization;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import dev.alvo.grad.AutoGradNode;
import dev.alvo.model.Matrix;
import dev.alvo.model.ModelParams;
import dev.alvo.model.Vector;
import dev.alvo.tokenizer.CodePointTokenizer;
import dev.alvo.tokenizer.Tokenizer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Serializes a {@link ModelData} to the <a href="https://github.com/huggingface/safetensors">
 * safetensors</a> binary format.
 *
 * <p>The layout is the standard one: an 8-byte little-endian {@code u64} header length, followed by
 * a UTF-8 JSON header, followed by the raw tensor buffers concatenated in the order their
 * {@code data_offsets} describe. Every weight matrix becomes a 2-D tensor; floating point values are
 * written little-endian as either {@code F32} (the default, for interoperability with PyTorch/NumPy)
 * or {@code F64} (for an exact round-trip of the model's {@code double} weights).
 *
 * <p>Safetensors stores tensors only, so everything that is not a tensor — the attention head count,
 * transformer block count, and the tokenizer's vocabulary — is carried in the header's
 * {@code __metadata__} string map, which is enough to fully reconstruct the {@link ModelParams} and
 * a {@link CodePointTokenizer} on the way back in.
 */
public final class SafeTensorsModelDataSerializer implements ModelDataSerializer<byte[]> {

  private static final String METADATA_KEY = "__metadata__";
  private static final String FORMAT = "format";
  private static final String FORMAT_VALUE = "jgpt";
  private static final String ATTENTION_HEAD_COUNT = "attention_head_count";
  private static final String TRANSFORMER_BLOCK_COUNT = "transformer_block_count";
  private static final String TOKENIZER_CLASS = "tokenizer_class";
  private static final String TOKENIZER_VOCABULARY = "tokenizer_vocabulary";

  private static final Gson GSON = new Gson();

  private final Dtype dtype;

  /**
   * The floating point encoding used for the tensor data buffers.
   */
  public enum Dtype {
    F32(4) {
      @Override
      void put(ByteBuffer buffer, double value) {
        buffer.putFloat((float) value);
      }

      @Override
      double get(ByteBuffer buffer) {
        return buffer.getFloat();
      }
    },
    F64(8) {
      @Override
      void put(ByteBuffer buffer, double value) {
        buffer.putDouble(value);
      }

      @Override
      double get(ByteBuffer buffer) {
        return buffer.getDouble();
      }
    };

    private final int byteSize;

    Dtype(int byteSize) {
      this.byteSize = byteSize;
    }

    abstract void put(ByteBuffer buffer, double value);

    abstract double get(ByteBuffer buffer);
  }

  /**
   * Creates a serializer that writes {@code F32} tensors (interoperable with PyTorch/NumPy).
   */
  public SafeTensorsModelDataSerializer() {
    this(Dtype.F32);
  }

  public SafeTensorsModelDataSerializer(Dtype dtype) {
    this.dtype = dtype;
  }

  @Override
  public byte[] serialize(ModelData data) {
    var params = data.params();
    var tensors = orderedTensors(params);

    var header = new JsonObject();
    header.add(METADATA_KEY, buildMetadata(params, data.tokenizer()));

    // Encode each tensor up front so the header can record exact, contiguous byte offsets.
    var chunks = new ArrayList<byte[]>(tensors.size());
    var offset = 0L;
    for (var tensor : tensors.entrySet()) {
      var matrix = tensor.getValue();
      var rows = matrix.vectors().size();
      var columns = rows == 0 ? 0 : matrix.get(0).nodes().size();
      var bytes = encodeMatrix(matrix);

      var shape = new JsonArray();
      shape.add(rows);
      shape.add(columns);

      var offsets = new JsonArray();
      offsets.add(offset);
      offsets.add(offset + bytes.length);

      var descriptor = new JsonObject();
      descriptor.addProperty("dtype", dtype.name());
      descriptor.add("shape", shape);
      descriptor.add("data_offsets", offsets);

      header.add(tensor.getKey(), descriptor);
      chunks.add(bytes);
      offset += bytes.length;
    }

    var headerBytes = GSON.toJson(header).getBytes(StandardCharsets.UTF_8);
    var output = ByteBuffer.allocate(Long.BYTES + headerBytes.length + (int) offset)
      .order(ByteOrder.LITTLE_ENDIAN);
    output.putLong(headerBytes.length);
    output.put(headerBytes);
    chunks.forEach(output::put);

    return output.array();
  }

  @Override
  public Optional<ModelData> deserialize(byte[] bytes) {
    try {
      var buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
      var headerLength = (int) buffer.getLong();
      var headerBytes = new byte[headerLength];
      buffer.get(headerBytes);
      var dataStart = Long.BYTES + headerLength;

      var header = GSON.fromJson(new String(headerBytes, StandardCharsets.UTF_8), JsonObject.class);
      var metadata = header.getAsJsonObject(METADATA_KEY);

      var blockCount = Integer.parseInt(metadata.get(TRANSFORMER_BLOCK_COUNT).getAsString());
      var headCount = Integer.parseInt(metadata.get(ATTENTION_HEAD_COUNT).getAsString());

      var tokenEmbeddings = readMatrix(header, bytes, dataStart, "token_embeddings");
      var positionEmbeddings = readMatrix(header, bytes, dataStart, "position_embeddings");

      var attentionQuery = new ArrayList<Matrix>(blockCount);
      var attentionKey = new ArrayList<Matrix>(blockCount);
      var attentionValue = new ArrayList<Matrix>(blockCount);
      var attentionOutput = new ArrayList<Matrix>(blockCount);
      var mlpFullyConnected1 = new ArrayList<Matrix>(blockCount);
      var mlpFullyConnected2 = new ArrayList<Matrix>(blockCount);
      for (var block = 0; block < blockCount; block++) {
        attentionQuery.add(readMatrix(header, bytes, dataStart, blockKey(block, "attn.query")));
        attentionKey.add(readMatrix(header, bytes, dataStart, blockKey(block, "attn.key")));
        attentionValue.add(readMatrix(header, bytes, dataStart, blockKey(block, "attn.value")));
        attentionOutput.add(readMatrix(header, bytes, dataStart, blockKey(block, "attn.output")));
        mlpFullyConnected1.add(readMatrix(header, bytes, dataStart, blockKey(block, "mlp.fc1")));
        mlpFullyConnected2.add(readMatrix(header, bytes, dataStart, blockKey(block, "mlp.fc2")));
      }

      var languageModelingHead = readMatrix(header, bytes, dataStart, "lm_head");

      var embeddingDimension = tokenEmbeddings.get(0).nodes().size();
      var headDimension = embeddingDimension / headCount;

      var params = new ModelParams(
        tokenEmbeddings,
        positionEmbeddings,
        attentionQuery,
        attentionKey,
        attentionValue,
        attentionOutput,
        mlpFullyConnected1,
        mlpFullyConnected2,
        languageModelingHead,
        headCount,
        headDimension,
        blockCount);

      return Optional.of(new ModelData(params, reconstructTokenizer(metadata)));
    } catch (RuntimeException e) {
      System.err.println("Error deserializing safetensors model: " + e.getMessage());
      return Optional.empty();
    }
  }

  private JsonObject buildMetadata(ModelParams params, Tokenizer tokenizer) {
    var metadata = new JsonObject();
    metadata.addProperty(FORMAT, FORMAT_VALUE);
    metadata.addProperty(ATTENTION_HEAD_COUNT, Integer.toString(params.attentionHeadCount()));
    metadata.addProperty(TRANSFORMER_BLOCK_COUNT, Integer.toString(params.transformerBlockCount()));
    metadata.addProperty(TOKENIZER_CLASS, tokenizer.getClass().getName());
    metadata.addProperty(TOKENIZER_VOCABULARY, tokenizer.getVocabulary().stream()
      .map(String::valueOf)
      .collect(Collectors.joining(",")));

    return metadata;
  }

  private byte[] encodeMatrix(Matrix matrix) {
    var rows = matrix.vectors().size();
    var columns = rows == 0 ? 0 : matrix.get(0).nodes().size();
    var buffer = ByteBuffer.allocate(rows * columns * dtype.byteSize).order(ByteOrder.LITTLE_ENDIAN);

    // Row-major (C order), matching how safetensors / PyTorch lay out a 2-D tensor.
    for (var vector : matrix.vectors()) {
      for (var node : vector.nodes()) {
        dtype.put(buffer, node.value());
      }
    }

    return buffer.array();
  }

  private Matrix readMatrix(JsonObject header, byte[] bytes, int dataStart, String name) {
    var descriptor = header.getAsJsonObject(name);
    if (descriptor == null) {
      throw new IllegalArgumentException("Missing tensor in safetensors header: " + name);
    }

    var tensorDtype = Dtype.valueOf(descriptor.get("dtype").getAsString());
    var shape = descriptor.getAsJsonArray("shape");
    var rows = shape.get(0).getAsInt();
    var columns = shape.size() > 1 ? shape.get(1).getAsInt() : 1;
    var start = descriptor.getAsJsonArray("data_offsets").get(0).getAsLong();

    var buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    buffer.position(dataStart + (int) start);

    var vectors = new ArrayList<Vector>(rows);
    for (var row = 0; row < rows; row++) {
      var nodes = new ArrayList<AutoGradNode>(columns);
      for (var column = 0; column < columns; column++) {
        nodes.add(new AutoGradNode(tensorDtype.get(buffer)));
      }
      vectors.add(new Vector(nodes));
    }

    return new Matrix(vectors);
  }

  private static Tokenizer reconstructTokenizer(JsonObject metadata) {
    var tokenizer = new CodePointTokenizer();
    var encoded = metadata.has(TOKENIZER_VOCABULARY)
      ? metadata.get(TOKENIZER_VOCABULARY).getAsString()
      : "";

    if (!encoded.isBlank()) {
      // CodePointTokenizer derives its vocabulary from the distinct, sorted code points of the
      // training documents. Feeding each stored code point back as a one-character document
      // therefore reproduces the original vocabulary exactly.
      var documents = Arrays.stream(encoded.split(","))
        .map(Integer::parseInt)
        .map(codePoint -> new String(Character.toChars(codePoint)))
        .toList();

      tokenizer.train(documents);
    }

    return tokenizer;
  }

  private static Map<String, Matrix> orderedTensors(ModelParams params) {
    var tensors = new LinkedHashMap<String, Matrix>();
    tensors.put("token_embeddings", params.tokenEmbeddings());
    tensors.put("position_embeddings", params.positionEmbeddings());

    for (var block = 0; block < params.transformerBlockCount(); block++) {
      tensors.put(blockKey(block, "attn.query"), params.attentionQuery().get(block));
      tensors.put(blockKey(block, "attn.key"), params.attentionKey().get(block));
      tensors.put(blockKey(block, "attn.value"), params.attentionValue().get(block));
      tensors.put(blockKey(block, "attn.output"), params.attentionOutput().get(block));
      tensors.put(blockKey(block, "mlp.fc1"), params.mlpFullyConnected1().get(block));
      tensors.put(blockKey(block, "mlp.fc2"), params.mlpFullyConnected2().get(block));
    }

    tensors.put("lm_head", params.languageModelingHead());
    return tensors;
  }

  private static String blockKey(int block, String suffix) {
    return "blocks." + block + "." + suffix;
  }
}
