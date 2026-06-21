package dev.alvo.model;

import static dev.alvo.grad.operation.GradNodeOperationDSL.div;

public final class Model {

  private static final double EPSILON = 1e-5;

  public Vector gpt(int tokenId, int positionId, ModelParams params, KVCache kvCache) {
    var tokenEmbeddings = params.tokenEmbeddings().get(tokenId);
    var positionEmbeddings = params.positionEmbeddings().get(positionId);
    var x = tokenEmbeddings.add(positionEmbeddings);

    // Transformer block
    for (int blockIndex = 0; blockIndex < params.transformerBlockCount(); blockIndex++) {
      var xResidual = x;
      x = x.rmsNorm(EPSILON);

      // Multi-head attention block
      var query = x.linear(params.attentionQuery().get(blockIndex));
      var key = x.linear(params.attentionKey().get(blockIndex));
      var value = x.linear(params.attentionValue().get(blockIndex));

      kvCache.append(blockIndex, key, value);


      // Attention head
      var attentionOutput = new Vector();
      for (int headIndex = 0; headIndex < params.attentionHeadCount(); headIndex++) {
        var headStartIndex = params.attentionHeadDimension() * headIndex;
        var headEndIndex = params.attentionHeadDimension() * (headIndex + 1);

        var queryHead = query.slice(headStartIndex, headEndIndex);

        var keyHead = kvCache.get(blockIndex).stream()
          .map(entry -> entry.key().slice(headStartIndex, headEndIndex))
          .toList();

        var valueHead = kvCache.get(blockIndex).stream()
          .map(entry -> entry.value().slice(headStartIndex, headEndIndex))
          .toList();

        var attentionLogits = keyHead.stream()
          .map(keyVector -> div(keyVector.dotProduct(queryHead), Math.sqrt(params.attentionHeadDimension()))) // scaled dot product
          .toList();

        var attentionProbabilities = new Vector(attentionLogits).softmax();

        for (int i = 0; i < params.attentionHeadDimension(); i++) {
          final int dim = i;
          var valueAttentionDimension = new Vector(valueHead.stream().map(v -> v.get(dim)).toList());
          attentionOutput.append(attentionProbabilities.dotProduct(valueAttentionDimension));
        }
      }

      x = attentionOutput.linear(params.attentionOutput().get(blockIndex));
      x = x.add(xResidual);

      // MLP block:
      xResidual = x;
      x = x.rmsNorm(EPSILON);
      x = x.linear(params.mlpFullyConnected1().get(blockIndex));
      x = x.relu();
      x = x.linear(params.mlpFullyConnected2().get(blockIndex));
      x = x.add(xResidual);
    }

    x = x.rmsNorm(EPSILON);
    x = x.linear(params.languageModelingHead());

    return x;
  }
}
