package com.komputation.layers

import jcuda.jcublas.cublasHandle
import com.komputation.cuda.CudaContext
import com.komputation.cuda.layers.forward.activation.CudaActivationLayer

interface CudaActivationLayerInstruction : CudaForwardLayerInstruction {

    override fun buildForCuda(context : CudaContext, cublasHandle : cublasHandle): CudaActivationLayer

}