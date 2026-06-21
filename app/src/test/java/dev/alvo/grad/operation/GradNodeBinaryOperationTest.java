package dev.alvo.grad.operation;

import dev.alvo.grad.AutoGradNode;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;

class GradNodeBinaryOperationTest {

  private static final double DELTA = 1e-9;

  private static AutoGradNode leaf(double value) {
    return new AutoGradNode(value, List.of(), List.of());
  }

  private static void assertChildGrads(AutoGradNode node, double... expected) {
    List<Double> actual = node.childGrads();
    assertEquals(expected.length, actual.size(), "child-grad count");
    for (int i = 0; i < expected.length; i++) {
      assertEquals(expected[i], actual.get(i), DELTA, "child-grad[" + i + "]");
    }
  }

  @Nested
  class Add {
    private final GradNodeBinaryOperation.Add op = new GradNodeBinaryOperation.Add();

    @Test
    void nodes() {
      AutoGradNode a = leaf(2d);
      AutoGradNode b = leaf(3d);
      AutoGradNode result = op.nodes(a, b);

      assertEquals(5d, result.value(), DELTA);
      assertChildGrads(result, 1d, 1d);
      assertIterableEquals(List.of(a, b), result.children());
    }

    @Test
    void left() {
      AutoGradNode a = leaf(2d);
      AutoGradNode result = op.left(a, 3d);

      assertEquals(5d, result.value(), DELTA);
      assertChildGrads(result, 1d);
      assertIterableEquals(List.of(a), result.children());
    }

    @Test
    void right() {
      AutoGradNode a = leaf(2d);
      AutoGradNode result = op.right(3d, a);

      assertEquals(5d, result.value(), DELTA);
      assertChildGrads(result, 1d);
      assertIterableEquals(List.of(a), result.children());
    }
  }

  @Nested
  class Sub {
    private final GradNodeBinaryOperation.Sub op = new GradNodeBinaryOperation.Sub();

    @Test
    void nodes() {
      AutoGradNode a = leaf(5d);
      AutoGradNode b = leaf(3d);
      AutoGradNode result = op.nodes(a, b);

      assertEquals(2d, result.value(), DELTA);
      assertChildGrads(result, 1d, -1d);
      assertIterableEquals(List.of(a, b), result.children());
    }

    @Test
    void left() {
      // f(a) = a - const, df/da = 1; child-grads must align 1:1 with the single child
      AutoGradNode a = leaf(5d);
      AutoGradNode result = op.left(a, 3d);

      assertEquals(2d, result.value(), DELTA);
      assertChildGrads(result, 1d);
      assertIterableEquals(List.of(a), result.children());
    }

    @Test
    void right() {
      // f(a) = const - a, df/da = -1
      AutoGradNode a = leaf(3d);
      AutoGradNode result = op.right(5d, a);

      assertEquals(2d, result.value(), DELTA);
      assertChildGrads(result, -1d);
      assertIterableEquals(List.of(a), result.children());
    }
  }

  @Nested
  class Mul {
    private final GradNodeBinaryOperation.Mul op = new GradNodeBinaryOperation.Mul();

    @Test
    void nodes() {
      AutoGradNode a = leaf(2d);
      AutoGradNode b = leaf(3d);
      AutoGradNode result = op.nodes(a, b);

      assertEquals(6d, result.value(), DELTA);
      // df/da = b, df/db = a
      assertChildGrads(result, 3d, 2d);
      assertIterableEquals(List.of(a, b), result.children());
    }

    @Test
    void left() {
      // f(a) = a * const, df/da = const (not the power rule)
      AutoGradNode a = leaf(2d);
      AutoGradNode result = op.left(a, 3d);

      assertEquals(6d, result.value(), DELTA);
      assertChildGrads(result, 3d);
      assertIterableEquals(List.of(a), result.children());
    }

    @Test
    void right() {
      // f(a) = const * a, df/da = const
      AutoGradNode a = leaf(2d);
      AutoGradNode result = op.right(3d, a);

      assertEquals(6d, result.value(), DELTA);
      assertChildGrads(result, 3d);
      assertIterableEquals(List.of(a), result.children());
    }
  }

  @Nested
  class Div {
    private final GradNodeBinaryOperation.Div op = new GradNodeBinaryOperation.Div();

    @Test
    void nodes() {
      AutoGradNode a = leaf(6d);
      AutoGradNode b = leaf(3d);
      AutoGradNode result = op.nodes(a, b);

      assertEquals(2d, result.value(), DELTA);
      // df/da = 1/b, df/db = -a/b^2
      assertChildGrads(result, 1d / 3d, -6d / 9d);
      assertIterableEquals(List.of(a, b), result.children());
    }

    @Test
    void left() {
      // f(a) = a / const, df/da = 1/const
      AutoGradNode a = leaf(6d);
      AutoGradNode result = op.left(a, 3d);

      assertEquals(2d, result.value(), DELTA);
      assertChildGrads(result, 1d / 3d);
      assertIterableEquals(List.of(a), result.children());
    }

    @Test
    void right() {
      // f(a) = const / a, df/da = -const/a^2
      AutoGradNode a = leaf(3d);
      AutoGradNode result = op.right(6d, a);

      assertEquals(2d, result.value(), DELTA);
      assertChildGrads(result, -6d / 9d);
      assertIterableEquals(List.of(a), result.children());
    }
  }

  @Nested
  class Pow {
    private final GradNodeBinaryOperation.Pow op = new GradNodeBinaryOperation.Pow();

    @Test
    void nodes() {
      AutoGradNode a = leaf(2d);
      AutoGradNode b = leaf(3d);
      AutoGradNode result = op.nodes(a, b);

      assertEquals(8d, result.value(), DELTA);
      // df/da = b * a^(b-1) = 3 * 2^2 = 12; df/db = a^b * ln(a) = 8 * ln(2)
      assertChildGrads(result, 12d, 8d * Math.log(2d));
      assertIterableEquals(List.of(a, b), result.children());
    }

    @Test
    void left() {
      // f(a) = a^const, df/da = const * a^(const-1)
      AutoGradNode a = leaf(2d);
      AutoGradNode result = op.left(a, 3d);

      assertEquals(8d, result.value(), DELTA);
      assertChildGrads(result, 3d * Math.pow(2d, 2d));
      assertIterableEquals(List.of(a), result.children());
    }

    @Test
    void right() {
      // f(a) = const^a, value = const^a, df/da = const^a * ln(const)
      AutoGradNode a = leaf(3d);
      AutoGradNode result = op.right(2d, a);

      assertEquals(8d, result.value(), DELTA);
      assertChildGrads(result, 8d * Math.log(2d));
      assertIterableEquals(List.of(a), result.children());
    }
  }
}
