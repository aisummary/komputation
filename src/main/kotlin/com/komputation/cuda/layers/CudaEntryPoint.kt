package com.komputation.cuda.layers

import jcuda.Pointer
import com.komputation.cuda.CudaForwardState
import com.komputation.cuda.memory.InputMemory
import com.komputation.matrix.Matrix

interface CudaEntryPoint : CudaForwardState {

    fun forward(
        batchId : Int,
        batchSize : Int,
        batch: IntArray,
        inputs: Array<Matrix>,
        memory : InputMemory) : Pointer

    fun backward(chain : Pointer) : Pointer

    val hasFixedLength : Boolean

}