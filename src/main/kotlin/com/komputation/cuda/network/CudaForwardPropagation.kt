package com.komputation.cuda.network

import jcuda.Pointer
import com.komputation.cuda.layers.CudaEntryPoint
import com.komputation.cuda.layers.CudaForwardLayer
import com.komputation.cuda.layers.CudaVariableLengthForwardLayer
import com.komputation.cuda.memory.InputMemory
import com.komputation.matrix.Matrix

interface CudaForwardPropagator {

    fun forward(batchId: Int, batchSize: Int, indices: IntArray, inputs: Array<Matrix>, memory : InputMemory, isTraining: Boolean) : Pointer

}

class CudaFixedLengthForwardPropagator(
    private val entryPoint: CudaEntryPoint,
    private val layers : Array<CudaForwardLayer>) : CudaForwardPropagator {

    override fun forward(batchId: Int, batchSize: Int, indices: IntArray, inputs: Array<Matrix>, memory : InputMemory, isTraining: Boolean) : Pointer {

        var result = this.entryPoint.forward(batchId, batchSize, indices, inputs, memory)

        for (layer in this.layers) {

            result = layer.forward(batchSize, result, isTraining)

        }

        return result

    }

}

class CudaVariableLengthForwardPropagator(
    private val entryPoint: CudaEntryPoint,
    layers : Array<CudaForwardLayer>) : CudaForwardPropagator {

    private val numberLayers = layers.size
    private val firstLayer = layers.first() as CudaVariableLengthForwardLayer
    private val remainingLayer = Array(this.numberLayers-1) { index -> layers[index+1] }

    override fun forward(batchId: Int, batchSize: Int, indices: IntArray, inputs: Array<Matrix>, memory : InputMemory, isTraining: Boolean) : Pointer {

        var result = this.entryPoint.forward(batchId, batchSize, indices, inputs, memory)

        result = this.firstLayer.forward(batchSize, memory.getLengths(batchId), result, isTraining)

        for (layer in this.remainingLayer) {

            result = layer.forward(batchSize, result, isTraining)

        }

        return result

    }

}

interface CudaChangesLengths {

    val deviceOutputLengths : Pointer

}