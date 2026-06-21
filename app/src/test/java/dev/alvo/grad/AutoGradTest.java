package dev.alvo.grad;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AutoGradTest {

  private static final double DELTA = 1e-9;

  private final AutoGrad autograd = new AutoGrad();

  /** Leaf (input) node: no children, no local derivatives. */
  private static AutoGradNode leaf(double value) {
    return new AutoGradNode(value, List.of(), List.of());
  }

  /**
   * Internal node carrying the local derivatives (childGrads[i] = d(this)/d(child[i]))
   * and the operands it was computed from.
   */
  private static AutoGradNode node(double value, List<Double> childGrads, List<AutoGradNode> children) {
    return new AutoGradNode(value, childGrads, children);
  }

  @Nested
  class Root {

    @Test
    void singleRootNodeGetsGradOne() {
      AutoGradNode root = leaf(5d);

      autograd.backpropagate(root);

      assertEquals(1d, root.grad(), DELTA);
    }

    @Test
    void rootGradIsOverwrittenWithOne() {
      // Root starts with a non-zero grad; backprop must seed it to exactly 1, not accumulate onto it.
      AutoGradNode root = new AutoGradNode(5d, 0.7d, List.of(), List.of());

      autograd.backpropagate(root);

      assertEquals(1d, root.grad(), DELTA);
    }
  }

  @Nested
  class SingleOperation {

    @Test
    void addPropagatesGradToBothOperands() {
      // f = a + b ; df/da = 1, df/db = 1
      AutoGradNode a = leaf(2d);
      AutoGradNode b = leaf(3d);
      AutoGradNode sum = node(5d, List.of(1d, 1d), List.of(a, b));

      autograd.backpropagate(sum);

      assertEquals(1d, sum.grad(), DELTA);
      assertEquals(1d, a.grad(), DELTA);
      assertEquals(1d, b.grad(), DELTA);
    }

    @Test
    void mulPropagatesUsingChainRule() {
      // f = a * b ; df/da = b, df/db = a
      AutoGradNode a = leaf(2d);
      AutoGradNode b = leaf(3d);
      AutoGradNode product = node(6d, List.of(b.value(), a.value()), List.of(a, b));

      autograd.backpropagate(product);

      assertEquals(1d, product.grad(), DELTA);
      assertEquals(3d, a.grad(), DELTA); // = b
      assertEquals(2d, b.grad(), DELTA); // = a
    }
  }

  @Nested
  class CompositeGraph {

    @Test
    void nestedExpressionAppliesChainRule() {
      // f = (a + b) * c, with a=2, b=3, c=4  ->  f = 20
      // df/da = c = 4, df/db = c = 4, df/dc = a + b = 5
      AutoGradNode a = leaf(2d);
      AutoGradNode b = leaf(3d);
      AutoGradNode c = leaf(4d);
      AutoGradNode sum = node(5d, List.of(1d, 1d), List.of(a, b));        // a + b
      AutoGradNode product = node(20d, List.of(c.value(), sum.value()), List.of(sum, c)); // sum * c

      autograd.backpropagate(product);

      assertEquals(1d, product.grad(), DELTA);
      assertEquals(4d, sum.grad(), DELTA);
      assertEquals(4d, a.grad(), DELTA);
      assertEquals(4d, b.grad(), DELTA);
      assertEquals(5d, c.grad(), DELTA);
    }

    @Test
    void deepChainMultipliesLocalDerivatives() {
      // a --(*3)--> x1 --(*5)--> x2 ; df/da = 3 * 5 = 15
      AutoGradNode a = leaf(2d);
      AutoGradNode x1 = node(6d, List.of(3d), List.of(a));
      AutoGradNode x2 = node(30d, List.of(5d), List.of(x1));

      autograd.backpropagate(x2);

      assertEquals(1d, x2.grad(), DELTA);
      assertEquals(5d, x1.grad(), DELTA);
      assertEquals(15d, a.grad(), DELTA);
    }
  }

  @Nested
  class SharedNodes {

    @Test
    void diamondAccumulatesGradientsFromAllPaths() {
      // a feeds two branches that are summed:
      //   b = a * 2, c = a * 3, d = b + c  ->  d = 5a, dd/da = 5
      // Correct accumulation requires `a` to be visited only after both b and c.
      AutoGradNode a = leaf(2d);
      AutoGradNode b = node(4d, List.of(2d), List.of(a));
      AutoGradNode c = node(6d, List.of(3d), List.of(a));
      AutoGradNode d = node(10d, List.of(1d, 1d), List.of(b, c));

      autograd.backpropagate(d);

      assertEquals(1d, d.grad(), DELTA);
      assertEquals(1d, b.grad(), DELTA);
      assertEquals(1d, c.grad(), DELTA);
      assertEquals(5d, a.grad(), DELTA); // 1*2 + 1*3
    }

    @Test
    void sameChildListedTwiceAccumulates() {
      // f = a * a, expressed as a node with `a` as both operands.
      // childGrads = [a, a]; df/da = 2a = 6
      AutoGradNode a = leaf(3d);
      AutoGradNode square = node(9d, List.of(a.value(), a.value()), List.of(a, a));

      autograd.backpropagate(square);

      assertEquals(1d, square.grad(), DELTA);
      assertEquals(6d, a.grad(), DELTA);
    }
  }

  @Nested
  class CycleDetection {

    @Test
    void cycleThrowsIllegalStateException() {
      // a -> b -> a ; the message must be cycle-safe (no StackOverflowError from node.toString()).
      AutoGradNode a = new AutoGradNode(1d, List.of(1d), List.of());
      AutoGradNode b = new AutoGradNode(1d, List.of(1d), List.of());
      a.setChildren(List.of(b));
      b.setChildren(List.of(a));

      IllegalStateException ex =
        assertThrows(IllegalStateException.class, () -> autograd.backpropagate(a));

      assertTrue(ex.getMessage().contains("Cycle detected"), ex.getMessage());
    }

    @Test
    void selfLoopThrowsIllegalStateException() {
      AutoGradNode a = new AutoGradNode(1d, List.of(1d), List.of());
      a.setChildren(List.of(a));

      assertThrows(IllegalStateException.class, () -> autograd.backpropagate(a));
    }
  }
}
