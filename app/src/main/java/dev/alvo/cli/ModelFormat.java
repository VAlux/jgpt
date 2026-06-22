package dev.alvo.cli;

import dev.alvo.repository.serialization.ByteArrayModelDataSerializer;
import dev.alvo.repository.serialization.JsonModelDataSerializer;
import dev.alvo.repository.serialization.ModelDataSerializer;
import dev.alvo.repository.serialization.SafeTensorsModelDataSerializer;

import java.util.function.Supplier;

/**
 * The on-disk model formats the CLI understands, each pairing a file extension with the serializer
 * that produces it. The {@code --type} option selects the format used when saving a freshly trained
 * model; {@link #fromExtension} maps a {@code --model} path back to its format when loading.
 *
 * <p>All three serializers produce {@code byte[]}, so they share a single
 * {@link dev.alvo.repository.storage.ByteArrayModelDataStorage} and the repository's type parameter
 * is always {@code byte[]} regardless of the chosen format.
 */
public enum ModelFormat {
  SAFETENSORS(".safetensors", SafeTensorsModelDataSerializer::new),
  JSON(".json", JsonModelDataSerializer::new),
  BIN(".bin", ByteArrayModelDataSerializer::new);

  private final String extension;
  private final Supplier<ModelDataSerializer<byte[]>> serializerFactory;

  ModelFormat(String extension, Supplier<ModelDataSerializer<byte[]>> serializerFactory) {
    this.extension = extension;
    this.serializerFactory = serializerFactory;
  }

  public String extension() {
    return this.extension;
  }

  public ModelDataSerializer<byte[]> serializer() {
    return this.serializerFactory.get();
  }

  /**
   * Resolves the format of an existing model file from its extension, so a {@code --model} path can
   * be loaded with the matching serializer.
   *
   * @throws IllegalArgumentException if the file has no recognized model extension
   */
  public static ModelFormat fromExtension(String fileName) {
    for (var format : values()) {
      if (fileName.toLowerCase().endsWith(format.extension)) {
        return format;
      }
    }

    throw new IllegalArgumentException(
      "Unknown model extension for '" + fileName + "'. Expected one of: "
        + SAFETENSORS.extension + ", " + JSON.extension + ", " + BIN.extension);
  }
}
