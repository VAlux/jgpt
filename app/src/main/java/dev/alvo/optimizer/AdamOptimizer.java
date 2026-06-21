package dev.alvo.optimizer;

import dev.alvo.grad.AutoGradNode;

import java.util.List;

public class AdamOptimizer {

  private final double learningRate;
  private final List<AutoGradNode> parameters;
  private final double beta1;
  private final double beta2;
  private final double epsilon;
  private int timestepIndex;
  private final double[] momentumM;
  private final double[] momentumV;

  public AdamOptimizer(double learningRate, List<AutoGradNode> parameters, double beta1, double beta2, double epsilon) {
    this.learningRate = learningRate;
    this.parameters = parameters;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    this.timestepIndex = 0;
    this.momentumM = new double[parameters.size()];
    this.momentumV = new double[parameters.size()];
  }

  public void step() {
    this.timestepIndex++;

    double mBiasCorrection = 1 - Math.pow(this.beta1, this.timestepIndex);
    double vBiasCorrection = 1 - Math.pow(this.beta2, this.timestepIndex);

    for (int paramIndex = 0; paramIndex < parameters.size(); paramIndex++) {
      var parameter = parameters.get(paramIndex);
      var grad = parameter.grad();

      this.momentumM[paramIndex] = this.beta1 * this.momentumM[paramIndex] + (1 - this.beta1) * grad;
      this.momentumV[paramIndex] = this.beta2 * this.momentumV[paramIndex] + (1 - this.beta2) * (grad * grad);

      var mHat = this.momentumM[paramIndex] / mBiasCorrection;
      var vHat = this.momentumV[paramIndex] / vBiasCorrection;

      parameter.setValue(parameter.value() - this.learningRate * mHat / (Math.sqrt(vHat) + this.epsilon));
      parameter.setGrad(0d);
    }
  }
}
