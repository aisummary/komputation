package com.komputation.cpu.layers.forward.normalization

import com.komputation.cpu.functions.activation.backwardNormalization
import com.komputation.cpu.functions.activation.normalize
import com.komputation.cpu.layers.BaseCpuVariableLengthForwardLayer
import com.komputation.cpu.layers.forward.activation.CpuActivationLayer

/*
    a/(a+b+c)

    input entry = a
    forward entry = a/(a+b+c)
 */
class CpuNormalizationLayer internal constructor(
    name : String? = null,
    numberRows : Int,
    minimumColumns : Int,
    maximumColumns : Int) : BaseCpuVariableLengthForwardLayer(name, numberRows, numberRows, minimumColumns, maximumColumns), CpuActivationLayer {

    private var sumsOverPossibleLengths = emptyArray<FloatArray>()
    private var sum = FloatArray(0)

    override fun acquire(maximumBatchSize: Int) {

        super.acquire(maximumBatchSize)

        this.sumsOverPossibleLengths = Array(this.numberLengths) { index -> FloatArray(this.numberInputRows * this.lengths[index]) }

    }

    override fun computeNumberOutputColumns(lengthIndex : Int, length: Int) = length

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean, result: FloatArray) {

        this.sum = this.sumsOverPossibleLengths[this.lengthIndex]

        normalize(this.numberInputRows, numberInputColumns, input, this.sum, this.forwardResult)

    }

    override fun computeBackwardResult(withinBatch: Int, chain: FloatArray, result: FloatArray) {

        backwardNormalization(this.numberInputRows, this.numberInputColumns, chain, this.forwardResult, this.sum, this.backwardResult)

    }

}