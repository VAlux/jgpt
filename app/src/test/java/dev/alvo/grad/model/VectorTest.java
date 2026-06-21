package dev.alvo.grad.model;

import dev.alvo.grad.AutoGrad;
import dev.alvo.grad.AutoGradNode;
import dev.alvo.model.Vector;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VectorTest {

  private static final double DELTA = 1e-9;

  private static AutoGradNode leaf(double value) {
    return new AutoGradNode(value, List.of(), List.of());
  }

  private static double[] values(Vector vector) {
    return vector.nodes().stream().mapToDouble(AutoGradNode::value).toArray();
  }

  /** Reference softmax computed independently of the production code, with the same max-shift. */
  private static double[] expectedSoftmax(double... xs) {
    double max = Arrays.stream(xs).max().orElse(0d);
    double[] exps = Arrays.stream(xs).map(x -> Math.exp(x - max)).toArray();
    double sum = Arrays.stream(exps).sum();
    return Arrays.stream(exps).map(e -> e / sum).toArray();
  }

  @Nested
  class Add {

    @Test
    void addsElementWise() {
      Vector a = new Vector(List.of(leaf(1d), leaf(2d), leaf(3d)));
      Vector b = new Vector(List.of(leaf(4d), leaf(5d), leaf(6d)));

      Vector result = a.add(b);

      assertEquals(3, result.nodes().size());
      assertArrayValues(result, 5d, 7d, 9d);
    }

    @Test
    void addsSingleElementVectors() {
      // A 1-element add is enough to catch the original set()/add() bug.
      Vector a = new Vector(List.of(leaf(2d)));
      Vector b = new Vector(List.of(leaf(3d)));

      Vector result = a.add(b);

      assertEquals(1, result.nodes().size());
      assertEquals(5d, result.nodes().get(0).value(), DELTA);
    }

    @Test
    void addsEmptyVectors() {
      Vector result = new Vector(List.of()).add(new Vector(List.of()));

      assertEquals(0, result.nodes().size());
    }

    @Test
    void wiresUpSumNodeChildrenAndGrads() {
      AutoGradNode left = leaf(1d);
      AutoGradNode right = leaf(4d);

      Vector result = new Vector(List.of(left)).add(new Vector(List.of(right)));
      AutoGradNode sum = result.nodes().get(0);

      // f(a, b) = a + b -> df/da = df/db = 1
      assertIterableEquals(List.of(left, right), sum.children());
      assertIterableEquals(List.of(1d, 1d), sum.childGrads());
    }

    @Test
    void doesNotMutateOperands() {
      AutoGradNode a0 = leaf(1d);
      AutoGradNode b0 = leaf(4d);
      Vector a = new Vector(List.of(a0));
      Vector b = new Vector(List.of(b0));

      a.add(b);

      assertEquals(1d, a.nodes().get(0).value(), DELTA);
      assertEquals(4d, b.nodes().get(0).value(), DELTA);
    }

    @Test
    void throwsWhenLeftIsLonger() {
      Vector a = new Vector(List.of(leaf(1d), leaf(2d)));
      Vector b = new Vector(List.of(leaf(1d)));

      IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> a.add(b));
      assertEquals("Vectors must have the same size", ex.getMessage());
    }

    @Test
    void throwsWhenRightIsLonger() {
      Vector a = new Vector(List.of(leaf(1d)));
      Vector b = new Vector(List.of(leaf(1d), leaf(2d)));

      assertThrows(IllegalArgumentException.class, () -> a.add(b));
    }

    private static void assertArrayValues(Vector vector, double... expected) {
      double[] actual = values(vector);
      assertEquals(expected.length, actual.length, "element count");
      for (int i = 0; i < expected.length; i++) {
        assertEquals(expected[i], actual[i], DELTA, "element[" + i + "]");
      }
    }
  }

  @Nested
  class RMSNorm {

    @Test
    void normalizesByRootMeanSquare() {
      // RMS([3, 4]) = sqrt((9 + 16) / 2) = sqrt(12.5) = 3.5355339...
      // -> [3 / rms, 4 / rms]
      Vector vector = new Vector(List.of(leaf(3d), leaf(4d)));

      Vector result = vector.rmsNorm(0d);

      double rms = Math.sqrt(12.5);
      double[] actual = values(result);
      assertEquals(2, actual.length);
      assertEquals(3d / rms, actual[0], DELTA);
      assertEquals(4d / rms, actual[1], DELTA);
    }

    @Test
    void isScaleInvariant() {
      // Scaling the input by a constant leaves the RMS-normalized output unchanged.
      Vector base = new Vector(List.of(leaf(1d), leaf(2d), leaf(3d)));
      Vector scaled = new Vector(List.of(leaf(10d), leaf(20d), leaf(30d)));

      double[] baseOut = values(base.rmsNorm(0d));
      double[] scaledOut = values(scaled.rmsNorm(0d));

      assertEquals(baseOut.length, scaledOut.length);
      for (int i = 0; i < baseOut.length; i++) {
        assertEquals(baseOut[i], scaledOut[i], DELTA, "element[" + i + "]");
      }
    }

    @Test
    void epsilonStabilizesZeroVector() {
      // With all-zero inputs the denominator is sqrt(0 + eps); outputs stay 0 (no NaN).
      Vector vector = new Vector(List.of(leaf(0d), leaf(0d)));

      double[] actual = values(vector.rmsNorm(1e-5));

      assertEquals(0d, actual[0], DELTA);
      assertEquals(0d, actual[1], DELTA);
    }

    @Test
    void throwsOnEmptyVector() {
      Vector empty = new Vector(List.of());

      assertThrows(IllegalArgumentException.class, () -> empty.rmsNorm(1e-5));
    }
  }

  @Nested
  class DotProduct {

    @Test
    void computesSumOfElementwiseProducts() {
      // [1, 2, 3] . [4, 5, 6] = 4 + 10 + 18 = 32
      Vector a = new Vector(List.of(leaf(1d), leaf(2d), leaf(3d)));
      Vector b = new Vector(List.of(leaf(4d), leaf(5d), leaf(6d)));

      AutoGradNode result = a.dotProduct(b);

      assertEquals(32d, result.value(), DELTA);
    }

    @Test
    void handlesNegativeValues() {
      // [1, -2] . [-3, 4] = -3 - 8 = -11
      Vector a = new Vector(List.of(leaf(1d), leaf(-2d)));
      Vector b = new Vector(List.of(leaf(-3d), leaf(4d)));

      assertEquals(-11d, a.dotProduct(b).value(), DELTA);
    }

    @Test
    void emptyVectorsDotToZero() {
      AutoGradNode result = new Vector(List.of()).dotProduct(new Vector(List.of()));

      assertEquals(0d, result.value(), DELTA);
      assertEquals(0, result.children().size());
    }

    @Test
    void wiresUpChildrenAndCrossGrads() {
      // result = a*b -> d/da = b, d/db = a. Each pair contributes both operands as children.
      AutoGradNode a0 = leaf(2d);
      AutoGradNode b0 = leaf(5d);
      AutoGradNode a1 = leaf(3d);
      AutoGradNode b1 = leaf(7d);

      AutoGradNode result = new Vector(List.of(a0, a1)).dotProduct(new Vector(List.of(b0, b1)));

      assertIterableEquals(List.of(a0, b0, a1, b1), result.children());
      assertIterableEquals(List.of(5d, 2d, 7d, 3d), result.childGrads());
    }

    @Test
    void backpropGivesEachOperandTheOtherAsGrad() {
      // d(a.b)/d(a_i) = b_i and d(a.b)/d(b_i) = a_i
      AutoGradNode a0 = leaf(2d);
      AutoGradNode a1 = leaf(3d);
      AutoGradNode b0 = leaf(5d);
      AutoGradNode b1 = leaf(7d);

      AutoGradNode result = new Vector(List.of(a0, a1)).dotProduct(new Vector(List.of(b0, b1)));
      new AutoGrad().backpropagate(result);

      assertEquals(5d, a0.grad(), DELTA);
      assertEquals(7d, a1.grad(), DELTA);
      assertEquals(2d, b0.grad(), DELTA);
      assertEquals(3d, b1.grad(), DELTA);
    }

    @Test
    void backpropAccumulatesGradForSelfDotProduct() {
      // x.x = sum(x_i^2) -> d/d(x_i) = 2*x_i. Same node appears twice per term.
      AutoGradNode x0 = leaf(3d);
      AutoGradNode x1 = leaf(4d);
      Vector x = new Vector(List.of(x0, x1));

      AutoGradNode result = x.dotProduct(x);
      new AutoGrad().backpropagate(result);

      assertEquals(25d, result.value(), DELTA); // 9 + 16
      assertEquals(6d, x0.grad(), DELTA);        // 2 * 3
      assertEquals(8d, x1.grad(), DELTA);        // 2 * 4
    }

    @Test
    void doesNotMutateOperands() {
      AutoGradNode a0 = leaf(1d);
      AutoGradNode b0 = leaf(4d);

      new Vector(List.of(a0)).dotProduct(new Vector(List.of(b0)));

      assertEquals(1d, a0.value(), DELTA);
      assertEquals(4d, b0.value(), DELTA);
      assertEquals(0d, a0.grad(), DELTA);
      assertEquals(0d, b0.grad(), DELTA);
    }

    @Test
    void throwsOnSizeMismatch() {
      Vector a = new Vector(List.of(leaf(1d), leaf(2d)));
      Vector b = new Vector(List.of(leaf(1d)));

      IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> a.dotProduct(b));
      assertEquals("Vectors must have the same size", ex.getMessage());
    }
  }

  @Nested
  class Softmax {

    @Test
    void matchesReferenceSoftmax() {
      Vector vector = new Vector(List.of(leaf(1d), leaf(2d), leaf(3d)));

      double[] actual = values(vector.softmax());
      double[] expected = expectedSoftmax(1d, 2d, 3d);

      assertEquals(3, actual.length);
      for (int i = 0; i < expected.length; i++) {
        assertEquals(expected[i], actual[i], DELTA, "element[" + i + "]");
      }
    }

    @Test
    void outputSumsToOne() {
      Vector vector = new Vector(List.of(leaf(-1d), leaf(0d), leaf(2d), leaf(5d)));

      double sum = Arrays.stream(values(vector.softmax())).sum();

      assertEquals(1d, sum, DELTA);
    }

    @Test
    void uniformInputGivesUniformDistribution() {
      Vector vector = new Vector(List.of(leaf(7d), leaf(7d), leaf(7d), leaf(7d)));

      double[] actual = values(vector.softmax());

      for (double p : actual) {
        assertEquals(0.25d, p, DELTA);
      }
    }

    @Test
    void singleElementMapsToOne() {
      Vector vector = new Vector(List.of(leaf(42d)));

      double[] actual = values(vector.softmax());

      assertEquals(1, actual.length);
      assertEquals(1d, actual[0], DELTA);
    }

    @Test
    void isStableForLargeInputs() {
      // Without the max-shift, exp(1000) overflows to Infinity and the result is NaN.
      Vector vector = new Vector(List.of(leaf(1000d), leaf(1001d), leaf(1002d)));

      double[] actual = values(vector.softmax());
      double[] expected = expectedSoftmax(0d, 1d, 2d); // shift-invariant

      for (int i = 0; i < expected.length; i++) {
        assertTrue(Double.isFinite(actual[i]), "element[" + i + "] should be finite");
        assertEquals(expected[i], actual[i], DELTA, "element[" + i + "]");
      }
    }

    @Test
    void backpropMatchesJacobian() {
      // d softmax_k / d x_j = s_k * (delta_kj - s_j). Backprop from a single output node.
      AutoGradNode x0 = leaf(1d);
      AutoGradNode x1 = leaf(2d);
      AutoGradNode x2 = leaf(3d);
      Vector vector = new Vector(List.of(x0, x1, x2));

      Vector softmax = vector.softmax();
      int k = 1; // differentiate output[1]
      new AutoGrad().backpropagate(softmax.nodes().get(k));

      double[] s = expectedSoftmax(1d, 2d, 3d);
      List<AutoGradNode> inputs = List.of(x0, x1, x2);
      for (int j = 0; j < inputs.size(); j++) {
        double expected = s[k] * ((k == j ? 1d : 0d) - s[j]);
        assertEquals(expected, inputs.get(j).grad(), DELTA, "grad[" + j + "]");
      }
    }

    @Test
    void doesNotMutateInput() {
      AutoGradNode x0 = leaf(1d);
      AutoGradNode x1 = leaf(2d);
      Vector vector = new Vector(List.of(x0, x1));

      vector.softmax();

      assertEquals(1d, x0.value(), DELTA);
      assertEquals(2d, x1.value(), DELTA);
      assertEquals(0d, x0.grad(), DELTA);
      assertEquals(0d, x1.grad(), DELTA);
    }
  }

  @Nested
  class LogSoftmax {

    @Test
    void equalsLogOfSoftmax() {
      Vector vector = new Vector(List.of(leaf(1d), leaf(2d), leaf(3d)));

      double[] actual = values(vector.logSoftmax());
      double[] expected = Arrays.stream(expectedSoftmax(1d, 2d, 3d)).map(Math::log).toArray();

      assertEquals(3, actual.length);
      for (int i = 0; i < expected.length; i++) {
        assertEquals(expected[i], actual[i], DELTA, "element[" + i + "]");
      }
    }

    @Test
    void exponentiatedOutputSumsToOne() {
      Vector vector = new Vector(List.of(leaf(-2d), leaf(0d), leaf(3d)));

      double sum = Arrays.stream(values(vector.logSoftmax())).map(Math::exp).sum();

      assertEquals(1d, sum, DELTA);
    }

    @Test
    void singleElementMapsToZero() {
      // log(softmax([x])) = log(1) = 0 for any single value.
      Vector vector = new Vector(List.of(leaf(42d)));

      double[] actual = values(vector.logSoftmax());

      assertEquals(1, actual.length);
      assertEquals(0d, actual[0], DELTA);
    }

    @Test
    void isStableForLargeInputs() {
      Vector vector = new Vector(List.of(leaf(1000d), leaf(1001d), leaf(1002d)));

      double[] actual = values(vector.logSoftmax());
      double[] expected = Arrays.stream(expectedSoftmax(0d, 1d, 2d)).map(Math::log).toArray();

      for (int i = 0; i < expected.length; i++) {
        assertTrue(Double.isFinite(actual[i]), "element[" + i + "] should be finite");
        assertEquals(expected[i], actual[i], DELTA, "element[" + i + "]");
      }
    }

    @Test
    void backpropMatchesJacobian() {
      // d logsoftmax_k / d x_j = delta_kj - s_j.
      AutoGradNode x0 = leaf(1d);
      AutoGradNode x1 = leaf(2d);
      AutoGradNode x2 = leaf(3d);
      Vector vector = new Vector(List.of(x0, x1, x2));

      Vector logSoftmax = vector.logSoftmax();
      int k = 2; // differentiate output[2]
      new AutoGrad().backpropagate(logSoftmax.nodes().get(k));

      double[] s = expectedSoftmax(1d, 2d, 3d);
      List<AutoGradNode> inputs = List.of(x0, x1, x2);
      for (int j = 0; j < inputs.size(); j++) {
        double expected = (k == j ? 1d : 0d) - s[j];
        assertEquals(expected, inputs.get(j).grad(), DELTA, "grad[" + j + "]");
      }
    }

    @Test
    void doesNotMutateInput() {
      AutoGradNode x0 = leaf(1d);
      AutoGradNode x1 = leaf(2d);
      Vector vector = new Vector(List.of(x0, x1));

      vector.logSoftmax();

      assertEquals(1d, x0.value(), DELTA);
      assertEquals(2d, x1.value(), DELTA);
      assertEquals(0d, x0.grad(), DELTA);
      assertEquals(0d, x1.grad(), DELTA);
    }
  }
}
