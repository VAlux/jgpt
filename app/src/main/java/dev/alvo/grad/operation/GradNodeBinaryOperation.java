package dev.alvo.grad.operation;

import dev.alvo.grad.AutoGradNode;

import java.util.List;

sealed interface GradNodeBinaryOperation extends GradNodeOperation {
  AutoGradNode nodes(AutoGradNode left, AutoGradNode right);

  AutoGradNode left(AutoGradNode left, double value);

  AutoGradNode right(double value, AutoGradNode right);

  final class Add implements GradNodeBinaryOperation {

    @Override
    public AutoGradNode nodes(AutoGradNode left, AutoGradNode right) {
      // f(a, b) = a + b
      // df/da = 1
      // df/db = 1

      return new AutoGradNode(
        left.value() + right.value(),
        List.of(1d, 1d),
        List.of(left, right));
    }

    @Override
    public AutoGradNode left(AutoGradNode left, double value) {
      return new AutoGradNode(
        left.value() + value,
        List.of(1d),
        List.of(left));
    }

    @Override
    public AutoGradNode right(double value, AutoGradNode right) {
      return this.left(right, value);
    }
  }

  final class Sub implements GradNodeBinaryOperation {
    @Override
    public AutoGradNode nodes(AutoGradNode left, AutoGradNode right) {
      // f(a, b) = a - b
      // df/da = 1
      // df/db = -1

      return new AutoGradNode(
        left.value() - right.value(),
        List.of(1d, -1d),
        List.of(left, right));
    }

    @Override
    public AutoGradNode left(AutoGradNode left, double value) {
      return new AutoGradNode(
        left.value() - value,
        List.of(1d),
        List.of(left));
    }

    @Override
    public AutoGradNode right(double value, AutoGradNode right) {
      return new AutoGradNode(
        value - right.value(),
        List.of(-1d),
        List.of(right));
    }
  }

  final class Mul implements GradNodeBinaryOperation {
    @Override
    public AutoGradNode nodes(AutoGradNode left, AutoGradNode right) {
      return new AutoGradNode(
        left.value() * right.value(),
        List.of(right.value(), left.value()),
        List.of(left, right)
      );
    }

    @Override
    public AutoGradNode left(AutoGradNode left, double value) {
      // f(a, b) = a * const
      // df/da = const

      return new AutoGradNode(
        left.value() * value,
        List.of(value),
        List.of(left)
      );
    }

    @Override
    public AutoGradNode right(double value, AutoGradNode right) {
      return this.left(right, value);
    }
  }

  final class Div implements GradNodeBinaryOperation {
    @Override
    public AutoGradNode nodes(AutoGradNode left, AutoGradNode right) {
      // f(a, b) = a / b
      // df/da = 1 / b
      // df/db = -a / b^2

      return new AutoGradNode(
        left.value() / right.value(),
        List.of(1d / right.value(), -left.value() / Math.pow(right.value(), 2)),
        List.of(left, right)
      );
    }

    @Override
    public AutoGradNode left(AutoGradNode left, double value) {
      // f(a, b) = a / const
      // df/da = 1 / const

      return new AutoGradNode(
        left.value() / value,
        List.of(1d / value),
        List.of(left)
      );
    }

    @Override
    public AutoGradNode right(double value, AutoGradNode right) {
      // f(a, b) = const / a
      // df/da = -const / a^2

      return new AutoGradNode(
        value / right.value(),
        List.of(-value / Math.pow(right.value(), 2)),
        List.of(right)
      );
    }
  }

  final class Pow implements GradNodeBinaryOperation {
    @Override
    public AutoGradNode nodes(AutoGradNode left, AutoGradNode right) {
      // f(a, b) = a^b
      // df/da = b * a^(b-1)
      // df/db = a^b * ln(a)

      double value = Math.pow(left.value(), right.value());

      return new AutoGradNode(
        value,
        List.of(
          right.value() * Math.pow(left.value(), right.value() - 1),
          value * Math.log(left.value())),
        List.of(left, right)
      );
    }

    @Override
    public AutoGradNode left(AutoGradNode left, double value) {
      // f(a, b) = a^b
      // df/da = b * a^(b-1)

      return new AutoGradNode(
        Math.pow(left.value(), value),
        List.of(value * Math.pow(left.value(), value - 1)),
        List.of(left)
      );
    }

    @Override
    public AutoGradNode right(double value, AutoGradNode right) {
      // f(a, b) = const^a
      // df/da = const^a * ln(const)

      return new AutoGradNode(
        Math.pow(value, right.value()),
        List.of(Math.pow(value, right.value()) * Math.log(value)),
        List.of(right)
      );
    }
  }
}
