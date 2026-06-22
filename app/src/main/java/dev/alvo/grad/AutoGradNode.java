package dev.alvo.grad;

import dev.alvo.grad.operation.GradNodeOperationDSL;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public final class AutoGradNode implements Serializable {

  @Serial
  private static final long serialVersionUID = 1L;

  private double value;
  private double grad;
  private double[] childGrads;
  private List<AutoGradNode> children;
  private GradNodeTopologicalLookupState topologicalState = GradNodeTopologicalLookupState.UNVISITED;

  public AutoGradNode() {
    this(0d);
  }

  public AutoGradNode(double value) {
    this(value, 0d, new double[0], List.of());
  }

  public AutoGradNode(double value, double grad, double[] childGrads, List<AutoGradNode> children) {
    this.value = value;
    this.grad = grad;
    this.childGrads = childGrads;
    this.children = children;
  }

  public AutoGradNode(double value, double[] childGrads, List<AutoGradNode> children) {
    this(value, 0d, childGrads, children);
  }

  public double value() {
    return this.value;
  }

  public void setValue(double value) {
    this.value = value;
  }

  public double grad() {
    return this.grad;
  }

  public void setGrad(double grad) {
    this.grad = grad;
  }

  public double[] childGrads() {
    return this.childGrads;
  }

  public void setChildGrads(double[] childGrads) {
    this.childGrads = childGrads;
  }

  public List<AutoGradNode> children() {
    return this.children;
  }

  public void setChildren(List<AutoGradNode> children) {
    this.children = children;
  }

  public GradNodeTopologicalLookupState topologicalState() {
    return this.topologicalState;
  }

  public void setTopologicalState(GradNodeTopologicalLookupState topologicalState) {
    this.topologicalState = topologicalState;
  }

  @Override
  public String toString() {
    return "AutoGradNode{" +
      "value=" + value +
      ", grad=" + grad +
      ", childGrads=" + Arrays.toString(childGrads) +
      ", children=" + children.size() +
      '}';
  }

  @Override
  public boolean equals(Object o) {
    if (o == null || getClass() != o.getClass()) return false;
    AutoGradNode that = (AutoGradNode) o;
    return Double.compare(value, that.value) == 0
      && Double.compare(grad, that.grad) == 0
      && Objects.deepEquals(childGrads, that.childGrads)
      && Objects.equals(children, that.children)
      && topologicalState == that.topologicalState;
  }

  @Override
  public int hashCode() {
    return Objects.hash(value, grad, Arrays.hashCode(childGrads), children, topologicalState);
  }

  public AutoGradNode relu() {
    return GradNodeOperationDSL.relu(this);
  }
}
