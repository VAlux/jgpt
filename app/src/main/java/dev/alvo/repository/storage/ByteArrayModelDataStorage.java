package dev.alvo.repository.storage;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Optional;

public class ByteArrayModelDataStorage implements ModelDataStorage<byte[]> {

  @Override
  public void save(Path path, String fileName, byte[] data) {
    File file = path.resolve(fileName).toFile();
    File parent = file.getParentFile();
    if (parent != null && !parent.exists()) {
      if (!parent.mkdirs()) {
        System.err.println("Error creating directory: " + parent);
      } else {
        System.out.println("Created directory: " + parent);
      }
    }

    System.out.println("Saving model to " + path);

    try (var fileOutputStream = new FileOutputStream(file)) {
      fileOutputStream.write(data);
    } catch (IOException e) {
      System.err.println("Error saving model: " + e.getMessage());
    }
  }

  @Override
  public Optional<byte[]> load(Path path) {
    var file = path.toFile();
    if (!file.exists()) {
      System.err.println("Model not found: " + path);
      return Optional.empty();
    }

    try (var fileInputStream = new FileInputStream(file);) {
      return Optional.of(fileInputStream.readAllBytes());
    } catch (IOException e) {
      System.err.println("Error loading model: " + e.getMessage());
      return Optional.empty();
    }
  }

  @Override
  public boolean exists(Path path) {
    return path.toFile().exists();
  }
}
