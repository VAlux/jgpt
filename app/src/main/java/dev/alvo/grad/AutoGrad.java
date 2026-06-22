package dev.alvo.grad;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static dev.alvo.grad.GradNodeTopologicalLookupState.IN_PROGRESS;
import static dev.alvo.grad.GradNodeTopologicalLookupState.VISITED;

public class AutoGrad {

  public void backpropagate(AutoGradNode root) {
    root.setGrad(1d);

    var ordered = getTopologicalOrder(root);
    for (var node : ordered.reversed()) {
      var children = node.children();
      for (int i = 0; i < children.size(); i++) {
        var child = children.get(i);
        child.setGrad(child.grad() + node.grad() * node.childGrads()[i]);
      }
    }
  }

  private List<AutoGradNode> getTopologicalOrder(AutoGradNode finalNode) {
    // @formatter:off
    record LookupState(AutoGradNode node, Iterator<AutoGradNode> children) {}
    // @formatter:on

    var results = new ArrayList<AutoGradNode>();
    var stack = new ArrayDeque<LookupState>();

    finalNode.setTopologicalState(IN_PROGRESS);
    stack.push(new LookupState(finalNode, finalNode.children().iterator()));

    while (!stack.isEmpty()) {
      var currentState = stack.peek();
      var currentNode = currentState.node();

      if (currentState.children().hasNext()) {
        var child = currentState.children().next();
        switch (child.topologicalState()) {
          case UNVISITED -> {
            child.setTopologicalState(IN_PROGRESS);
            stack.push(new LookupState(child, child.children().iterator()));
          }
          case IN_PROGRESS -> throw new IllegalStateException(
            "Cycle detected for node value=%s (id=%s)".formatted(
              child.value(), System.identityHashCode(child)));
        }
      } else {
        currentNode.setTopologicalState(VISITED);
        results.add(currentNode);
        stack.pop();
      }
    }

    return results;
  }
}
