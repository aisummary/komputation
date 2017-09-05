package com.komputation.cuda.optimization

import jcuda.Pointer

interface CudaUpdateRule {

    fun denseUpdate(
        count : Int,
        pointerToParameters: Pointer,
        pointerToGradient: Pointer)

    fun sparseUpdate(
        maximumParameters : Int,
        pointerToIndices: Pointer,
        pointerToCounts: Pointer,
        pointerToParameters: Pointer,
        pointerToGradient: Pointer)


}