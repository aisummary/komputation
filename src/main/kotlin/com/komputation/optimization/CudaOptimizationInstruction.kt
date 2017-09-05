package com.komputation.optimization

import com.komputation.cuda.CudaContext
import com.komputation.cuda.optimization.CudaOptimizationStrategy

interface CudaOptimizationInstruction {

    fun buildForCuda(context: CudaContext) : CudaOptimizationStrategy

}