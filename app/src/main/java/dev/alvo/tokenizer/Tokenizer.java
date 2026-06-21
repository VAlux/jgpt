package dev.alvo.tokenizer;

import java.io.Serializable;
import java.util.List;
import java.util.Set;

public interface Tokenizer extends Serializable {
  void train(List<String> documents);

  int[] encode(String text);

  String decode(int[] tokenIds);

  Integer bosToken();

  Integer eosToken();

  Integer unkToken();

  List<Integer> getVocabulary();

  Integer getVocabularySize();

  default Set<Integer> specialTokens() {
    return Set.of(bosToken(), eosToken(), unkToken());
  }
}
