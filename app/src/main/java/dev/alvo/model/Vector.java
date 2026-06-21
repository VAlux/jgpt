package dev.alvo.model;

import dev.alvo.grad.AutoGradNode;
import dev.alvo.grad.operation.GradNodeOperationDSL;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import static dev.alvo.grad.operation.GradNodeOperationDSL.div;
import static dev.alvo.grad.operation.GradNodeOperationDSL.log;
import static dev.alvo.grad.operation.GradNodeOperationDSL.pow;
import static dev.alvo.grad.operation.GradNodeOperationDSL.sub;
import static dev.alvo.grad.operation.GradNodeOperationDSL.sum;

/**
 * An immutable mathematical vector whose components are {@link AutoGradNode}s, allowing every
 * operation performed on the vector to participate in reverse-mode automatic differentiation.
 *
 * <p>Because each component carries its own gradient information, operations such as
 * {@link #add(Vector)}, {@link #dotProduct(Vector)} and {@link #rmsNorm(double)} build up a
 * computation graph that can later be back-propagated through. Instances are immutable: every
 * operation returns a new {@code Vector} (or {@link AutoGradNode}) rather than mutating the
 * receiver.
 *
 * @param nodes the ordered components of the vector; the size of this list defines the
 *              dimensionality of the vector
 */
public record Vector(List<AutoGradNode> nodes) implements Serializable {

  public Vector() {
    this(new ArrayList<>());
  }

  /**
   * Performs element-wise addition of this vector and {@code other}, returning a new vector whose
   * {@code i}-th component is the sum of the corresponding components of the two operands.
   *
   * @param other the vector to add to this one; must have the same size as this vector
   * @return a new {@code Vector} representing the element-wise sum
   * @throws IllegalArgumentException if the vectors have different sizes
   */
  public Vector add(Vector other) {
    if (this.nodes.size() != other.nodes.size()) {
      throw new IllegalArgumentException("Vectors must have the same size");
    }

    var result = new ArrayList<AutoGradNode>(nodes.size());
    for (var i = 0; i < this.nodes.size(); i++) {
      result.add(GradNodeOperationDSL.add(this.nodes.get(i), other.nodes.get(i)));
    }

    return new Vector(result);
  }

  /**
   * Applies Root Mean Square Layer Normalization (RMSNorm) to this vector.
   *
   * <p>Each component is divided by the root mean square of all components, i.e.
   * {@code x_i / sqrt(mean(x_j^2) + epsilon)}. Unlike standard layer normalization, RMSNorm does
   * not subtract the mean, only rescaling the vector by its magnitude. The small {@code epsilon}
   * term is added under the square root to avoid division by zero and to improve numerical
   * stability for near-zero vectors.
   *
   * @param epsilon a small positive constant added for numerical stability
   * @return a new {@code Vector} containing the normalized components
   * @throws IllegalArgumentException if this vector is empty
   */
  public Vector rmsNorm(double epsilon) {
    if (this.nodes.isEmpty()) {
      throw new IllegalArgumentException("Cannot RMSNorm an empty vector");
    }

    var sqSum = sum(this.nodes.stream().map(GradNodeOperationDSL::sq).toList());
    var meanSq = div(sqSum, this.nodes.size());
    var denominator = pow(GradNodeOperationDSL.add(meanSq, epsilon), 0.5);

    return new Vector(this.nodes.stream().map(node -> div(node, denominator)).toList());
  }

  /**
   * Computes the dot product of this vector with {@code other}, returning a single
   * {@link AutoGradNode} that is wired into the autograd graph.
   *
   * <p>The result is {@code sum(a_i * b_i)}. Rather than composing the product and sum operations
   * from the DSL, this method builds the resulting node directly for efficiency: it records each
   * operand as a child together with its local gradient. Since {@code d(result)/d(a_i) = b_i} and
   * {@code d(result)/d(b_i) = a_i}, the value of the opposite operand is stored as the local
   * gradient for each child.
   *
   * @param other the vector to take the dot product with; must have the same size as this vector
   * @return an {@code AutoGradNode} holding the scalar dot product and its gradient wiring
   * @throws IllegalArgumentException if the vectors have different sizes
   */
  public AutoGradNode dotProduct(Vector other) {
    if (this.nodes.size() != other.nodes.size()) {
      throw new IllegalArgumentException("Vectors must have the same size");
    }

    var size = this.nodes.size();
    var value = 0d;
    var children = new ArrayList<AutoGradNode>(size * 2);
    var childGrads = new ArrayList<Double>(size * 2);

    for (int i = 0; i < size; i++) {
      var currentNodes = this.nodes.get(i);
      var otherNodes = other.nodes.get(i);

      // result = sum(a_i * b_i)  =>  d/d(a_i) = b_i, d/d(b_i) = a_i
      value += currentNodes.value() * otherNodes.value();

      children.add(currentNodes);
      children.add(otherNodes);

      childGrads.add(otherNodes.value());
      childGrads.add(currentNodes.value());
    }

    return new AutoGradNode(value, childGrads, children);
  }

  /**
   * Multiplies this vector by the given {@code matrix}, producing a new vector.
   *
   * <p>Each component of the result is the dot product of this vector with one of the matrix's row
   * vectors, so the resulting vector has one component per row in {@code matrix}. This is the core
   * operation of a fully connected (linear) layer.
   *
   * @param matrix the matrix to multiply by; each of its vectors must match the size of this vector
   * @return a new {@code Vector} whose components are the dot products with each row of the matrix
   */
  public Vector mul(Matrix matrix) {
    return new Vector(matrix.vectors().stream().map(this::dotProduct).toList());
  }

  /**
   * Applies a linear transformation defined by {@code matrix} to this vector.
   *
   * <p>This is a semantic alias for {@link #mul(Matrix)}, named to reflect its use as a linear
   * (fully connected) layer in a neural network.
   *
   * @param matrix the weight matrix defining the linear transformation
   * @return a new {@code Vector} holding the transformed result
   */
  public Vector linear(Matrix matrix) {
    return mul(matrix);
  }

  public Vector slice(int headStartIndex, int headEndIndex) {
    return new Vector(this.nodes.subList(headStartIndex, headEndIndex));
  }

  public AutoGradNode get(int index) {
    return this.nodes.get(index);
  }

  public void append(AutoGradNode node) {
    this.nodes.add(node);
  }

  public Vector relu() {
    return new Vector(this.nodes.stream().map(AutoGradNode::relu).toList());
  }

  public Vector softmax() {
    var maxValue = this.nodes.stream().mapToDouble(AutoGradNode::value).max().orElse(0d);
    var shifted = this.nodes.stream().map(node -> sub(node, maxValue)).toList();
    var exponents = shifted.stream().map(GradNodeOperationDSL::exp).toList();
    var sumExps = sum(exponents);

    return new Vector(exponents.stream().map(exp -> div(exp, sumExps)).toList());
  }

  public Vector logSoftmax() {
    var maxValue = this.nodes.stream().mapToDouble(AutoGradNode::value).max().orElse(0d);
    var shifted = this.nodes.stream().map(node -> sub(node, maxValue)).toList();
    var exps = shifted.stream().map(GradNodeOperationDSL::exp).toList();
    var logSumExps = log(sum(exps));

    return new Vector(shifted.stream().map(elem -> sub(elem, logSumExps)).toList());
  }
}
