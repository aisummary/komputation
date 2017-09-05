package com.komputation.cuda.kernels

object ForwardKernels {

    fun bias() = KernelInstruction(
        "biasKernel",
        "biasKernel",
        "forward/bias/BiasKernel.cu",
        listOf(KernelHeaders.nan))

    fun dropoutTraining() = KernelInstruction(
        "dropoutTrainingKernel",
        "dropoutTrainingKernel",
        "forward/dropout/DropoutTrainingKernel.cu",
        listOf(KernelHeaders.nan))

    fun dropoutRuntime() = KernelInstruction(
        "dropoutRuntimeKernel",
        "dropoutRuntimeKernel",
        "forward/dropout/DropoutRuntimeKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardDropout() = KernelInstruction(
        "backwardDropoutKernel",
        "backwardDropoutKernel",
        "forward/dropout/BackwardDropoutKernel.cu",
        listOf(KernelHeaders.nan))

    fun exponentiation() = KernelInstruction(
        "exponentiationKernel",
        "exponentiationKernel",
        "forward/exponentiation/ExponentiationKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardExponentiation() = KernelInstruction(
        "backwardExponentiationKernel",
        "backwardExponentiationKernel",
        "forward/exponentiation/BackwardExponentiationKernel.cu",
        listOf(KernelHeaders.nan))

    fun normalization() = KernelInstruction(
        "normalizationKernel",
        "normalizationKernel",
        "forward/normalization/NormalizationKernel.cu",
        listOf(KernelHeaders.sumReduction, KernelHeaders.nan))

    fun backwardNormalization() = KernelInstruction(
        "backwardNormalizationKernel",
        "backwardNormalizationKernel",
        "forward/normalization/BackwardNormalizationKernel.cu",
        listOf(KernelHeaders.sumReduction, KernelHeaders.nan))

    fun sigmoid() = KernelInstruction(
        "sigmoidKernel",
        "sigmoidKernel",
        "forward/sigmoid/SigmoidKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardSigmoid() = KernelInstruction(
        "backwardSigmoidKernel",
        "backwardSigmoidKernel",
        "forward/sigmoid/BackwardSigmoidKernel.cu",
        listOf(KernelHeaders.nan))

    fun relu() = KernelInstruction(
        "reluKernel",
        "reluKernel",
        "forward/relu/ReluKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardRelu() = KernelInstruction(
        "backwardReluKernel",
        "backwardReluKernel",
        "forward/relu/BackwardReluKernel.cu",
        listOf(KernelHeaders.nan))

    fun tanh() = KernelInstruction("tanhKernel",
        "tanhKernel",
        "forward/tanh/TanhKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardTanh() = KernelInstruction(
        "backwardTanhKernel",
        "backwardTanhKernel",
        "forward/tanh/BackwardTanhKernel.cu",
        listOf(KernelHeaders.nan))

    fun maxPooling() = KernelInstruction(
        "maxPoolingKernel",
        "maxPoolingKernel",
        "forward/maxpooling/MaxPoolingKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardMaxPooling() = KernelInstruction(
        "backwardMaxPoolingKernel",
        "backwardMaxPoolingKernel",
        "forward/maxpooling/BackwardMaxPoolingKernel.cu")

    fun expansion() = KernelInstruction(
        "expansionKernel",
        "expansionKernel",
        "forward/expansion/ExpansionKernel.cu",
        listOf(KernelHeaders.nan, KernelHeaders.zero))

    fun backwardExpansion() = KernelInstruction(
        "backwardExpansionKernel",
        "backwardExpansionKernel",
        "forward/expansion/BackwardExpansionKernel.cu",
        listOf(KernelHeaders.sumReduction))

}