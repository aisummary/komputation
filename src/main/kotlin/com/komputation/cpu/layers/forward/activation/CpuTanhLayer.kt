package com.komputation.cpu.layers.forward.activation

import com.komputation.cpu.functions.activation.differentiateTanh
import com.komputation.cpu.functions.activation.tanh
import com.komputation.cpu.functions.hadamard
import com.komputation.cpu.layers.BaseCpuVariableLengthForwardLayer

class CpuTanhLayer internal constructor(
    name: String? = null,
    numberRows : Int,
    minimumColumns : Int,
    maximumColumns : Int) : BaseCpuVariableLengthForwardLayer(name, numberRows, numberRows, minimumColumns, maximumColumns), CpuActivationLayer {

    private var hasCachedDifferentiation = false
    private var differentiationsOverPossibleLengths = emptyArray<FloatArray>()
    private var differentiation = FloatArray(0)

    private var numberInputEntries = -1

    override fun acquire(maximumBatchSize: Int) {

        super.acquire(maximumBatchSize)

        this.differentiationsOverPossibleLengths = Array(this.numberLengths) { index -> FloatArray(this.numberInputRows * this.lengths[index]) }

    }

    override fun computeNumberOutputColumns(lengthIndex : Int, length: Int) = length

    override fun forward(withinBatch : Int, numberInputColumns : Int, input : FloatArray, isTraining : Boolean): FloatArray {

        super.forward(withinBatch, numberInputColumns, input, isTraining)

        this.hasCachedDifferentiation = false

        return this.forwardResult

    }


    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean, result: FloatArray) {

        this.numberInputEntries = input.size

        tanh(input, this.forwardResult, this.numberInputEntries)

    }

    override fun backward(withinBatch : Int, chain : FloatArray): FloatArray {

        this.differentiation = this.differentiationsOverPossibleLengths[this.lengthIndex]

        if (!this.hasCachedDifferentiation) {

            differentiateTanh(this.forwardResult, this.differentiation, this.numberInputEntries)

            this.hasCachedDifferentiation = true

        }

        super.backward(withinBatch, chain)

        return this.backwardResult

    }

    override fun computeBackwardResult(withinBatch: Int, chain: FloatArray, result: FloatArray) {

        hadamard(chain, this.differentiation, this.backwardResult, this.numberInputEntries)

    }

}