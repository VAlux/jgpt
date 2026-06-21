package dev.alvo.repository.storage;

import java.nio.file.Path;
import java.util.Optional;

public interface ModelDataStorage<T> {
  void save(Path path, String fileName, T data);

  Optional<T> load(Path path);

  boolean exists(Path path);
}

