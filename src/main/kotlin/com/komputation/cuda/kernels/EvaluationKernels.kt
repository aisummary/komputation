package com.komputation.cuda.kernels

object EvaluationKernels {

    fun evaluation() = KernelInstruction(
        "evaluationKernel",
        "evaluationKernel",
        "evaluation/EvaluationKernel.cu")

}