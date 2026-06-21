package dev.alvo.repository;

import dev.alvo.repository.serialization.ModelDataSerializer;
import dev.alvo.repository.serialization.ModelDataSerializer.ModelData;
import dev.alvo.repository.storage.ModelDataStorage;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashSet;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

public class ModelDataRepository<T> implements DataRepository {

  private final ModelDataSerializer<T> serializer;
  private final ModelDataStorage<T> storage;
  private final Path path;
  private final ModelNameBuilder nameBuilder;

  public ModelDataRepository(String path,
                             ModelDataSerializer<T> serializer,
                             ModelDataStorage<T> storage,
                             ModelNameBuilder nameBuilder) {

    this.serializer = serializer;
    this.storage = storage;
    this.path = Paths.get(path);
    this.nameBuilder = nameBuilder;
  }

  @Override
  public void save(ModelData data) {
    storage.save(this.path, nameBuilder.build(data.params(), ""), serializer.serialize(data));
  }

  @Override
  public void save(ModelData data, String suffix) {
    storage.save(this.path, nameBuilder.build(data.params(), suffix), serializer.serialize(data));
  }

  @Override
  public Optional<ModelData> load(String fileName) {
    return resolveExisting(fileName)
      .flatMap(storage::load)
      .flatMap(serializer::deserialize);
  }

  @Override
  public boolean exists(String fileName) {
    return resolveExisting(fileName).isPresent();
  }

  /**
   * Returns the human-readable, absolute locations {@link #load} searches for {@code fileName},
   * so callers can build an actionable error message when nothing is found.
   */
  public String searchedLocations(String fileName) {
    return candidates(fileName).stream()
      .map(candidate -> candidate.toAbsolutePath().toString())
      .collect(Collectors.joining(", "));
  }

  private Optional<Path> resolveExisting(String fileName) {
    return candidates(fileName).stream()
      .filter(this.storage::exists)
      .findFirst();
  }

  /**
   * A model is looked up first as given (absolute, or relative to the working directory) and then
   * inside the repository's directory, so callers can pass either a bare checkpoint name or an
   * explicit path regardless of where the process was started from.
   */
  private Set<Path> candidates(String fileName) {
    var given = Paths.get(fileName);
    var candidates = new LinkedHashSet<Path>();
    candidates.add(given);
    candidates.add(this.path.resolve(given));
    return candidates;
  }
}
