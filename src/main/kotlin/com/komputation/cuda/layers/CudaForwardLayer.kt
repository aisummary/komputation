package com.komputation.cuda.layers

import jcuda.Pointer
import com.komputation.cuda.CudaLayerState

interface CudaForwardLayer : CudaLayerState {

    fun forward(batchSize: Int, deviceInput: Pointer, isTraining: Boolean): Pointer

    fun backward(batchSize: Int, chain: Pointer) : Pointer

}

interface CudaVariableLengthForwardLayer {

    fun forward(batchSize: Int, deviceLengths: Pointer, deviceInput: Pointer, isTraining: Boolean): Pointer

}