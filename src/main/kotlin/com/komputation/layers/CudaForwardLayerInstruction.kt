package com.komputation.layers

import jcuda.jcublas.cublasHandle
import com.komputation.cuda.CudaContext
import com.komputation.cuda.layers.CudaForwardLayer

interface CudaForwardLayerInstruction {

    fun buildForCuda(context : CudaContext, cublasHandle : cublasHandle) : CudaForwardLayer

}