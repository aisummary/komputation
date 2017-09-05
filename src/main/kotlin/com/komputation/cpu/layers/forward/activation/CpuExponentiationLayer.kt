package com.komputation.cpu.layers.forward.activation

import com.komputation.cpu.functions.activation.backwardExponentiation
import com.komputation.cpu.functions.activation.exponentiate
import com.komputation.cpu.layers.BaseCpuVariableLengthForwardLayer

class CpuExponentiationLayer internal constructor(
    name : String? = null,
    numberRows : Int,
    minimumColumns : Int,
    maximumColumns : Int) : BaseCpuVariableLengthForwardLayer(name, numberRows, numberRows, minimumColumns, maximumColumns), CpuActivationLayer {

    private var numberEntries = -1

    override fun computeNumberOutputColumns(lengthIndex : Int, length: Int) = length

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean, result: FloatArray) {

        this.numberEntries = input.size

        exponentiate(input, this.forwardResult, this.numberEntries)

    }

    override fun computeBackwardResult(withinBatch: Int, chain: FloatArray, result: FloatArray) {

        backwardExponentiation(this.forwardResult, chain, result, this.numberEntries)

    }

}