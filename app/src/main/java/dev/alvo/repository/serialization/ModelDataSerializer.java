package dev.alvo.repository.serialization;

import dev.alvo.model.ModelParams;
import dev.alvo.tokenizer.Tokenizer;

import java.io.Serializable;
import java.util.Optional;

public interface ModelDataSerializer<T> {
  record ModelData(ModelParams params, Tokenizer tokenizer) implements Serializable {}

  T serialize(ModelData data);

  Optional<ModelData> deserialize(T data);
}
