package dev.alvo.grad.operation;

sealed interface GradNodeOperation permits GradNodeBinaryOperation, GradNodeUnaryOperation {
}

