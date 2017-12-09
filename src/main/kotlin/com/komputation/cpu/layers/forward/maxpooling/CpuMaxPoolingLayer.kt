package com.komputation.cpu.layers.forward.maxpooling

import com.komputation.cpu.functions.findMaxIndicesInRows
import com.komputation.cpu.functions.selectEntries
import com.komputation.cpu.layers.BaseCpuVariableLengthForwardLayer
import java.util.*

class CpuMaxPoolingLayer internal constructor(
    name : String? = null,
    numberInputRows : Int,
    minimumInputColumns : Int,
    maximumInputColumns : Int) : BaseCpuVariableLengthForwardLayer(name, numberInputRows, numberInputRows, minimumInputColumns, maximumInputColumns) {

    private val maxRowIndices = IntArray(this.numberInputRows)

    override fun computeNumberOutputColumns(inputLength: Int) = 1

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean, forwardResult: FloatArray) {
        findMaxIndicesInRows(input, this.numberInputRows, numberInputColumns, this.maxRowIndices)

        selectEntries(input, this.maxRowIndices, forwardResult, this.numberInputRows)
    }

    override fun computeBackwardResult(withinBatch: Int, numberInputColumns: Int, numberOutputColumns : Int, forwardResult: FloatArray, chain: FloatArray, backwardResult: FloatArray) {
        Arrays.fill(backwardResult, 0f)

        for (indexRow in 0 until this.numberInputRows) {
            backwardResult[this.maxRowIndices[indexRow]] = chain[indexRow]
        }
    }

}