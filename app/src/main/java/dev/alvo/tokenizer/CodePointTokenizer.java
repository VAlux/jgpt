package dev.alvo.tokenizer;

import java.io.Serial;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class CodePointTokenizer implements Tokenizer {

  @Serial
  private static final long serialVersionUID = 1L;

  private final List<Integer> vocabulary;
  private final Map<Integer, Integer> tokenToTokenId;

  public CodePointTokenizer() {
    this.vocabulary = new ArrayList<>();
    this.tokenToTokenId = new HashMap<>();
  }

  @Override
  public void train(List<String> documents) {
    this.vocabulary.clear();
    this.tokenToTokenId.clear();

    documents.stream()
      .flatMapToInt(String::codePoints)
      .distinct()
      .sorted()
      .forEach(vocabulary::add);

    for (int index = 0; index < vocabulary.size(); index++) {
      this.tokenToTokenId.put(vocabulary.get(index), index);
    }
  }

  @Override
  public int[] encode(String text) {
    if (text == null || text.isEmpty()) {
      return new int[]{bosToken(), eosToken()};
    }

    var encoded = text.chars()
      .map(character -> this.tokenToTokenId.getOrDefault(character, this.unkToken()))
      .toArray();

    var sequence = new int[encoded.length + 2];
    sequence[0] = bosToken();
    System.arraycopy(encoded, 0, sequence, 1, encoded.length);
    sequence[sequence.length - 1] = eosToken();

    return sequence;
  }

  @Override
  public String decode(int[] tokenIds) {
    Set<Integer> specialTokens = this.specialTokens();

    return Arrays.stream(tokenIds)
      .filter(id -> !specialTokens.contains(id))
      .mapToObj(id -> new String(Character.toChars(vocabulary.get(id))))
      .collect(Collectors.joining(""));
  }

  @Override
  public Integer bosToken() {
    return this.vocabulary.size();
  }

  @Override
  public Integer eosToken() {
    return this.vocabulary.size() + 1;
  }

  @Override
  public Integer unkToken() {
    return this.vocabulary.size() + 2;
  }

  @Override
  public List<Integer> getVocabulary() {
    return vocabulary;
  }

  @Override
  public Integer getVocabularySize() {
    return this.vocabulary.size() + specialTokens().size();
  }
}
