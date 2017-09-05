package com.komputation.layers

import com.komputation.cuda.CudaContext
import com.komputation.cuda.layers.CudaEntryPoint

interface CudaEntryPointInstruction {

    fun buildForCuda(context : CudaContext) : CudaEntryPoint

}