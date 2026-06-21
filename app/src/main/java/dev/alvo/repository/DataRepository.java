package dev.alvo.repository;

import dev.alvo.repository.serialization.ModelDataSerializer.ModelData;

import java.util.Optional;

public interface DataRepository {
  void save(ModelData data);

  void save(ModelData data, String suffix);

  Optional<ModelData> load(String fileName);

  boolean exists(String path);
}
