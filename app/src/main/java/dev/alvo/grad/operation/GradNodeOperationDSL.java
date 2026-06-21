package dev.alvo.grad.operation;

import dev.alvo.grad.AutoGradNode;
import dev.alvo.grad.operation.GradNodeBinaryOperation.Add;
import dev.alvo.grad.operation.GradNodeBinaryOperation.Div;
import dev.alvo.grad.operation.GradNodeBinaryOperation.Mul;
import dev.alvo.grad.operation.GradNodeBinaryOperation.Pow;
import dev.alvo.grad.operation.GradNodeBinaryOperation.Sub;
import dev.alvo.grad.operation.GradNodeUnaryOperation.Exp;
import dev.alvo.grad.operation.GradNodeUnaryOperation.Log;
import dev.alvo.grad.operation.GradNodeUnaryOperation.Neg;
import dev.alvo.grad.operation.GradNodeUnaryOperation.ReLU;

import java.util.Collections;
import java.util.List;

public class GradNodeOperationDSL {

  private static final Add add;
  private static final Sub sub;
  private static final Mul mul;
  private static final Div div;
  private static final Pow pow;
  private static final Neg neg;
  private static final Log log;
  private static final Exp exp;
  private static final ReLU reLU;

  static {
    add = new Add();
    sub = new Sub();
    mul = new Mul();
    div = new Div();
    pow = new Pow();
    neg = new Neg();
    log = new Log();
    exp = new Exp();
    reLU = new ReLU();
  }

  public static AutoGradNode node(double value) {
    return new AutoGradNode(value);
  }

  public static AutoGradNode sum(List<AutoGradNode> nodes) {
    var sum = nodes.stream().mapToDouble(AutoGradNode::value).sum();
    var grads = Collections.nCopies(nodes.size(), 1d);

    return new AutoGradNode(sum, grads, nodes);
  }

  public static AutoGradNode add(AutoGradNode left, AutoGradNode right) {
    return add.nodes(left, right);
  }

  public static AutoGradNode add(AutoGradNode left, double right) {
    return add.left(left, right);
  }

  public static AutoGradNode add(double left, AutoGradNode right) {
    return add.right(left, right);
  }

  public static AutoGradNode sub(AutoGradNode left, AutoGradNode right) {
    return sub.nodes(left, right);
  }

  public static AutoGradNode sub(AutoGradNode left, double right) {
    return sub.left(left, right);
  }

  public static AutoGradNode sub(double left, AutoGradNode right) {
    return sub.right(left, right);
  }

  public static AutoGradNode mul(AutoGradNode left, AutoGradNode right) {
    return mul.nodes(left, right);
  }

  public static AutoGradNode mul(AutoGradNode left, double right) {
    return mul.left(left, right);
  }

  public static AutoGradNode mul(double left, AutoGradNode right) {
    return mul.right(left, right);
  }

  public static AutoGradNode sq(AutoGradNode node) {
    return mul.nodes(node, node);
  }

  public static AutoGradNode div(AutoGradNode left, AutoGradNode right) {
    return div.nodes(left, right);
  }

  public static AutoGradNode div(AutoGradNode left, double right) {
    return div.left(left, right);
  }

  public static AutoGradNode div(double left, AutoGradNode right) {
    return div.right(left, right);
  }

  public static AutoGradNode pow(AutoGradNode left, AutoGradNode right) {
    return pow.nodes(left, right);
  }

  public static AutoGradNode pow(AutoGradNode left, double right) {
    return pow.left(left, right);
  }

  public static AutoGradNode pow(double left, AutoGradNode right) {
    return pow.right(left, right);
  }

  public static AutoGradNode neg(AutoGradNode node) {
    return neg.nodes(node);
  }

  public static AutoGradNode log(AutoGradNode node) {
    return log.nodes(node);
  }

  public static AutoGradNode exp(AutoGradNode node) {
    return exp.nodes(node);
  }

  public static AutoGradNode relu(AutoGradNode node) {
    return reLU.nodes(node);
  }
}
