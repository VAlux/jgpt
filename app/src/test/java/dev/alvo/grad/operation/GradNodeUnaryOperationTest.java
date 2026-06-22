package dev.alvo.grad.operation;

import dev.alvo.grad.AutoGradNode;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;

class GradNodeUnaryOperationTest {

  private static final double DELTA = 1e-9;

  private static AutoGradNode leaf(double value) {
    return new AutoGradNode(value, new double[0], List.of());
  }

  private static void assertChildGrads(AutoGradNode node, double... expected) {
    double[] actual = node.childGrads();
    assertEquals(expected.length, actual.length, "child-grad count");
    for (int i = 0; i < expected.length; i++) {
      assertEquals(expected[i], actual[i], DELTA, "child-grad[" + i + "]");
    }
  }

  @Nested
  class Neg {
    private final GradNodeUnaryOperation.Neg op = new GradNodeUnaryOperation.Neg();

    @Test
    void positiveValue() {
      // f(x) = -x, df/dx = -1
      AutoGradNode x = leaf(3d);
      AutoGradNode result = op.nodes(x);

      assertEquals(-3d, result.value(), DELTA);
      assertChildGrads(result, -1d);
      assertIterableEquals(List.of(x), result.children());
    }

    @Test
    void negativeValue() {
      // gradient is constant regardless of input sign
      AutoGradNode x = leaf(-5d);
      AutoGradNode result = op.nodes(x);

      assertEquals(5d, result.value(), DELTA);
      assertChildGrads(result, -1d);
      assertIterableEquals(List.of(x), result.children());
    }
  }

  @Nested
  class Log {
    private final GradNodeUnaryOperation.Log op = new GradNodeUnaryOperation.Log();

    @Test
    void nodes() {
      // f(x) = ln(x), df/dx = 1/x
      AutoGradNode x = leaf(2d);
      AutoGradNode result = op.nodes(x);

      assertEquals(Math.log(2d), result.value(), DELTA);
      assertChildGrads(result, 1d / 2d);
      assertIterableEquals(List.of(x), result.children());
    }

    @Test
    void logOfOneIsZero() {
      // ln(1) = 0, df/dx = 1/1 = 1
      AutoGradNode x = leaf(1d);
      AutoGradNode result = op.nodes(x);

      assertEquals(0d, result.value(), DELTA);
      assertChildGrads(result, 1d);
      assertIterableEquals(List.of(x), result.children());
    }
  }

  @Nested
  class Exp {
    private final GradNodeUnaryOperation.Exp op = new GradNodeUnaryOperation.Exp();

    @Test
    void nodes() {
      // f(x) = e^x, value == df/dx == e^x
      AutoGradNode x = leaf(2d);
      AutoGradNode result = op.nodes(x);

      assertEquals(Math.exp(2d), result.value(), DELTA);
      assertChildGrads(result, Math.exp(2d));
      assertIterableEquals(List.of(x), result.children());
    }

    @Test
    void expOfZeroIsOne() {
      // e^0 = 1, df/dx = e^0 = 1
      AutoGradNode x = leaf(0d);
      AutoGradNode result = op.nodes(x);

      assertEquals(1d, result.value(), DELTA);
      assertChildGrads(result, 1d);
      assertIterableEquals(List.of(x), result.children());
    }
  }
}
