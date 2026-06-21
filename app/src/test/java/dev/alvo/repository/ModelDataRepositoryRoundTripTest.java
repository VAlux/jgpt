package dev.alvo.repository;

import dev.alvo.model.ModelParams;
import dev.alvo.repository.serialization.ByteArrayModelDataSerializer;
import dev.alvo.repository.serialization.ModelDataSerializer.ModelData;
import dev.alvo.repository.storage.ByteArrayModelDataStorage;
import dev.alvo.tokenizer.CodePointTokenizer;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelDataRepositoryRoundTripTest {

  @Test
  void savesAndLoadsByExactFileName(@org.junit.jupiter.api.io.TempDir Path tmp) {
    var nameBuilder = new ModelNameBuilder();
    var repository = new ModelDataRepository<>(
      tmp.toString(),
      new ByteArrayModelDataSerializer(),
      new ByteArrayModelDataStorage(),
      nameBuilder);

    var tokenizer = new CodePointTokenizer();
    tokenizer.train(List.of("kyiv", "lviv"));
    var params = ModelParams.create(tokenizer.getVocabularySize(), 16, 20, 1, 4, 4, new Random(1));
    var data = new ModelData(params, tokenizer);

    repository.save(data);

    var fileName = nameBuilder.build(params, "");
    assertEquals("jgpt-vocab-8-seq-20-emb-16-trans-1-attn-4.bin", fileName,
      "un-suffixed model name must not contain a stray trailing delimiter");

    Optional<ModelData> loaded = repository.load(fileName);
    assertTrue(loaded.isPresent(), "model should load by the name the builder produced");
    assertEquals(params.getVocabularySize(), loaded.get().params().getVocabularySize());
  }

  @Test
  void loadingMissingFileReturnsEmpty(@org.junit.jupiter.api.io.TempDir Path tmp) {
    var repository = new ModelDataRepository<>(
      tmp.toString(),
      new ByteArrayModelDataSerializer(),
      new ByteArrayModelDataStorage(),
      new ModelNameBuilder());

    assertTrue(repository.load("does-not-exist.bin").isEmpty());
  }

  @Test
  void loadsViaExplicitPathRegardlessOfRepositoryDirectory(@org.junit.jupiter.api.io.TempDir Path tmp) {
    // Repository points at a *different* directory than where the file actually lives;
    // an explicit/absolute path must still resolve, independent of the working directory.
    var storeDir = tmp.resolve("checkpoints");
    var repository = new ModelDataRepository<>(
      tmp.resolve("somewhere-else").toString(),
      new ByteArrayModelDataSerializer(),
      new ByteArrayModelDataStorage(),
      new ModelNameBuilder());

    var tokenizer = new CodePointTokenizer();
    tokenizer.train(List.of("kyiv", "lviv"));
    var params = ModelParams.create(tokenizer.getVocabularySize(), 16, 20, 1, 4, 4, new Random(1));

    var saver = new ModelDataRepository<>(
      storeDir.toString(),
      new ByteArrayModelDataSerializer(),
      new ByteArrayModelDataStorage(),
      new ModelNameBuilder());
    saver.save(new ModelData(params, tokenizer), "iter-1");

    var absolutePath = storeDir.resolve(new ModelNameBuilder().build(params, "iter-1")).toString();
    assertTrue(repository.load(absolutePath).isPresent(),
      "an explicit absolute path should load even when the repo dir does not contain the file");
  }

  @Test
  void searchedLocationsAreAbsoluteAndActionable(@org.junit.jupiter.api.io.TempDir Path tmp) {
    var repository = new ModelDataRepository<>(
      tmp.resolve("checkpoints").toString(),
      new ByteArrayModelDataSerializer(),
      new ByteArrayModelDataStorage(),
      new ModelNameBuilder());

    var searched = repository.searchedLocations("model.bin");
    assertTrue(searched.contains(tmp.resolve("checkpoints").resolve("model.bin").toString()),
      "diagnostic should include the absolute checkpoints location: " + searched);
  }
}
