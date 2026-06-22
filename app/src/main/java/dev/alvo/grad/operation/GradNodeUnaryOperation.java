package dev.alvo.grad.operation;

import dev.alvo.grad.AutoGradNode;

import java.util.List;

sealed interface GradNodeUnaryOperation extends GradNodeOperation {
  AutoGradNode nodes(AutoGradNode node);

  final class Neg implements GradNodeUnaryOperation {

    @Override
    public AutoGradNode nodes(AutoGradNode node) {
      return new AutoGradNode(
        -node.value(),
        new float[]{-1f},
        List.of(node));
    }
  }

  final class Log implements GradNodeUnaryOperation {

    @Override
    public AutoGradNode nodes(AutoGradNode node) {
      return new AutoGradNode(
        Math.log(node.value()),
        new float[]{(float) (1d / node.value())},
        List.of(node)
      );
    }
  }

  final class Exp implements GradNodeUnaryOperation {
    @Override
    public AutoGradNode nodes(AutoGradNode node) {
      double exp = Math.exp(node.value());

      return new AutoGradNode(
        exp,
        new float[]{(float) exp},
        List.of(node)
      );
    }
  }

  final class ReLU implements GradNodeUnaryOperation {

    @Override
    public AutoGradNode nodes(AutoGradNode node) {
      return new AutoGradNode(
        Math.max(0, node.value()),
        new float[]{node.value() > 0 ? 1f : 0f},
        List.of(node)
      );
    }
  }
}
