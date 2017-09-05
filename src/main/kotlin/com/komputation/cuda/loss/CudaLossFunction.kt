package com.komputation.cuda.loss

import jcuda.Pointer
import com.komputation.layers.Resourceful

interface CudaLossFunction : Resourceful {

    fun accumulate(pointerToPredictions: Pointer, pointerToTargets : Pointer, batchSize: Int)

    fun accessAccumulation() : Float

    fun backward(pointerToPredictions: Pointer, pointerToTargets : Pointer, batchSize : Int): Pointer

}