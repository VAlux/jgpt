package dev.alvo.repository;

import dev.alvo.model.ModelParams;

import java.util.ArrayList;
import java.util.List;

public class ModelNameBuilder {

  private static final String DELIMITER = "-";
  private static final String PREFIX = "jgpt";
  private static final String EXTENSION = ".bin";

  public String build(ModelParams params, String suffix) {
    var parts = new ArrayList<>(List.of(
      PREFIX,
      "vocab-" + params.getVocabularySize(),
      "seq-" + params.getMaxSequenceLength(),
      "emb-" + params.getEmbeddingDimension(),
      "trans-" + params.transformerBlockCount(),
      "attn-" + params.attentionHeadCount()
    ));

    if (suffix != null && !suffix.isBlank()) {
      parts.add(suffix);
    }

    return String.join(DELIMITER, parts) + EXTENSION;
  }
}
