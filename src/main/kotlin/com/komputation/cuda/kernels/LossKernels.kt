package com.komputation.cuda.kernels

object LossKernels {

    fun squaredLoss() =
        KernelInstruction(
            "squaredLossKernel",
            "squaredLossKernel",
            "loss/squaredloss/SquaredLossKernel.cu",
            listOf(KernelHeaders.sumReduction))

    fun backwardSquaredLoss() =
        KernelInstruction(
            "backwardSquaredLossKernel",
            "backwardSquaredLossKernel",
            "loss/squaredloss/BackwardSquaredLossKernel.cu",
            listOf(KernelHeaders.nan))

    fun logisticLoss(blockSize: Int) =
        KernelInstruction(
            "logisticLossKernel",
            "logisticLossKernel<$blockSize>",
            "loss/logisticloss/LogisticLossKernel.cu",
            listOf(KernelHeaders.sumReduction))

    fun backwardLogisticLoss() =
        KernelInstruction(
            "backwardLogisticLossKernel",
            "backwardLogisticLossKernel",
            "loss/logisticloss/BackwardLogisticLossKernel.cu",
            listOf(KernelHeaders.nan))

}