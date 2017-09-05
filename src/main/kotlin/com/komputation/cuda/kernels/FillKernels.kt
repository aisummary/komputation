package com.komputation.cuda.kernels

object FillKernels {

    fun oneIntegerArray() = KernelInstruction(
        "fillOneIntegerArrayKernel",
        "fillOneIntegerArrayKernel",
        "fill/FillOneIntegerArrayKernel.cu")

    fun twoIntegerArrays() = KernelInstruction(
        "fillTwoIntegerArraysKernel",
        "fillTwoIntegerArraysKernel",
        "fill/FillTwoIntegerArraysKernel.cu")

    fun oneFloatArray() = KernelInstruction(
        "fillOneFloatArrayKernel",
        "fillOneFloatArrayKernel",
        "fill/FillOneFloatArrayKernel.cu")

    fun twoFloatArrays() = KernelInstruction(
        "fillTwoFloatArraysKernel",
        "fillTwoFloatArraysKernel",
        "fill/FillTwoFloatArraysKernel.cu")

}