package dev.alvo.optimizer;

import dev.alvo.grad.AutoGradNode;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AdamOptimizerTest {

  // Parameters are stored as float (~1e-7 precision), so assertions use a float-scale tolerance.
  private static final double DELTA = 1e-5;

  private static final double BETA1 = 0.9d;
  private static final double BETA2 = 0.999d;
  private static final double EPSILON = 1e-8d;

  /** A parameter node seeded with a value and a pending gradient. */
  private static AutoGradNode param(double value, double grad) {
    return new AutoGradNode(value, grad, new float[0], List.of());
  }

  @Nested
  class SingleStep {

    @Test
    void firstStepMovesParameterAgainstGradient() {
      // On t=1 the bias correction cancels the (1-beta) factors, so:
      //   mHat = g, vHat = g^2, update = lr * g / (|g| + eps)
      double lr = 0.1d;
      double value = 1.0d;
      double grad = 0.5d;
      AutoGradNode p = param(value, grad);
      var optimizer = new AdamOptimizer(lr, List.of(p), BETA1, BETA2, EPSILON);

      optimizer.step();

      double expectedUpdate = lr * grad / (Math.sqrt(grad * grad) + EPSILON);
      assertEquals(value - expectedUpdate, p.value(), DELTA);
    }

    @Test
    void negativeGradientMovesParameterUp() {
      double lr = 0.1d;
      double value = 1.0d;
      double grad = -0.5d;
      AutoGradNode p = param(value, grad);
      var optimizer = new AdamOptimizer(lr, List.of(p), BETA1, BETA2, EPSILON);

      optimizer.step();

      assertTrue(p.value() > value, "negative gradient should increase the parameter");
    }

    @Test
    void gradientIsResetToZeroAfterStep() {
      AutoGradNode p = param(1.0d, 0.5d);
      var optimizer = new AdamOptimizer(0.1d, List.of(p), BETA1, BETA2, EPSILON);

      optimizer.step();

      assertEquals(0d, p.grad(), DELTA);
    }

    @Test
    void zeroGradientLeavesParameterUnchanged() {
      AutoGradNode p = param(1.0d, 0d);
      var optimizer = new AdamOptimizer(0.1d, List.of(p), BETA1, BETA2, EPSILON);

      optimizer.step();

      // m and v stay 0, so the update is 0 / eps = 0.
      assertEquals(1.0d, p.value(), DELTA);
    }
  }

  @Nested
  class MultipleParameters {

    @Test
    void parametersAreUpdatedIndependently() {
      AutoGradNode a = param(1.0d, 0.5d);
      AutoGradNode b = param(2.0d, -0.5d);
      var optimizer = new AdamOptimizer(0.1d, List.of(a, b), BETA1, BETA2, EPSILON);

      optimizer.step();

      assertTrue(a.value() < 1.0d, "a had a positive gradient and should decrease");
      assertTrue(b.value() > 2.0d, "b had a negative gradient and should increase");
    }
  }

  @Nested
  class BiasCorrection {

    @Test
    void biasCorrectionKeepsConstantGradientStepStable() {
      // With a constant gradient, bias correction makes m̂_t = g and v̂_t = g² for every t, so the
      // update lr·m̂/(√v̂ + ε) is the same at every step. The point of bias correction is precisely
      // that the very first step is already a full lr-scaled step rather than a suppressed one.
      double lr = 0.1d;
      double grad = 0.5d;

      AutoGradNode p = param(0d, grad);
      var optimizer = new AdamOptimizer(lr, List.of(p), BETA1, BETA2, EPSILON);

      optimizer.step();
      double afterFirst = p.value();
      double firstDisplacement = Math.abs(afterFirst - 0d);

      p.setGrad(grad);
      optimizer.step();
      double secondDisplacement = Math.abs(p.value() - afterFirst);

      // Bias correction prevents a tiny first step: displacement ≈ lr from step one.
      assertEquals(lr, firstDisplacement, 1e-3);
      // And the per-step displacement stays constant under a constant gradient.
      assertEquals(firstDisplacement, secondDisplacement, DELTA);
    }
  }

  @Nested
  class Convergence {

    @Test
    void minimizesQuadraticTowardTarget() {
      // Minimize f(x) = (x - target)^2, whose gradient is 2 * (x - target).
      double target = 3.0d;
      AutoGradNode x = param(0d, 0d);
      var optimizer = new AdamOptimizer(0.05d, List.of(x), BETA1, BETA2, EPSILON);

      for (int i = 0; i < 5000; i++) {
        x.setGrad(2 * (x.value() - target));
        optimizer.step();
      }

      assertEquals(target, x.value(), 1e-2);
    }
  }
}
