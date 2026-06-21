package dev.alvo.repository.serialization;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Optional;

public final class ByteArrayModelDataSerializer implements ModelDataSerializer<byte[]> {

  @Override
  public byte[] serialize(ModelData data) {

    var outputStream = new ByteArrayOutputStream();
    try (var objectOutputStream = new ObjectOutputStream(outputStream)) {
      objectOutputStream.writeObject(data);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    return outputStream.toByteArray();
  }

  @Override
  public Optional<ModelData> deserialize(byte[] bytes) {
    ModelData result;

    try (var objectInputStream = new ObjectInputStream(new ByteArrayInputStream(bytes))) {
      result = (ModelData) objectInputStream.readObject();
    } catch (IOException | ClassNotFoundException e) {
      System.err.println("Error deserializing model: " + e.getMessage());
      return Optional.empty();
    }

    return Optional.of(result);
  }
}
