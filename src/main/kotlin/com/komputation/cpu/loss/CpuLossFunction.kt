package com.komputation.cpu.loss

import com.komputation.cpu.layers.CpuBackwardState


interface CpuLossFunction : CpuBackwardState {

    fun forward(predictions: FloatArray, targets : FloatArray): Float

    fun backward(predictions: FloatArray, targets : FloatArray) : FloatArray

}