package dev.alvo.model;

import java.util.ArrayList;
import java.util.List;

public final class KVCache {

  private final List<List<KVCacheEntry>> entries;

  public record KVCacheEntry(Vector key, Vector value) {}

  public KVCache(int blockCount) {
    this.entries = new ArrayList<>(blockCount);
    for (int i = 0; i < blockCount; i++) {
      this.entries.add(new ArrayList<>());
    }
  }

  public void append(int index, KVCacheEntry entry) {
    this.entries.get(index).add(entry);
  }

  public void append(int index, Vector key, Vector value) {
    this.append(index, new KVCacheEntry(key, value));
  }

  public List<KVCacheEntry> get(int blockIndex) {
    return this.entries.get(blockIndex);
  }
}
