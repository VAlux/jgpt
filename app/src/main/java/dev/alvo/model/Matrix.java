package dev.alvo.model;

import dev.alvo.grad.AutoGradNode;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

public record Matrix(List<Vector> vectors) implements Serializable {

  public Vector get(int index) {
    return vectors.get(index);
  }

  public static Matrix random(int rows, int cols, double stddev, Random random) {
    var vectors = new ArrayList<Vector>();

    for (int i = 0; i < rows; i++) {
      var vector = new Vector();
      for (int j = 0; j < cols; j++) {
        vector.append(new AutoGradNode(random.nextGaussian(0d, stddev)));
      }

      vectors.add(vector);
    }

    return new Matrix(vectors);
  }

  public Stream<AutoGradNode> nodes() {
    return this.vectors.stream().flatMap(vector -> vector.nodes().stream());
  }
}
