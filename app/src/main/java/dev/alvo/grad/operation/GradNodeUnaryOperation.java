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
        List.of(-1d),
        List.of(node));
    }
  }

  final class Log implements GradNodeUnaryOperation {

    @Override
    public AutoGradNode nodes(AutoGradNode node) {
      return new AutoGradNode(
        Math.log(node.value()),
        List.of(1d / node.value()),
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
        List.of(exp),
        List.of(node)
      );
    }
  }

  final class ReLU implements GradNodeUnaryOperation {

    @Override
    public AutoGradNode nodes(AutoGradNode node) {
      return new AutoGradNode(
        Math.max(0, node.value()),
        List.of(node.value() > 0 ? 1d : 0d),
        List.of(node)
      );
    }
  }
}
