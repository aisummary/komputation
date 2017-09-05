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

    private var maxRowIndices = IntArray(0)

    override fun acquire(maximumBatchSize: Int) {

        super.acquire(maximumBatchSize)

        this.maxRowIndices = IntArray(this.numberInputRows)

    }

    override fun computeNumberOutputColumns(lengthIndex : Int, length: Int) = 1

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean, result: FloatArray) {

        findMaxIndicesInRows(input, this.numberInputRows, numberInputColumns, this.maxRowIndices)

        selectEntries(input, this.maxRowIndices, result, this.numberInputRows)

    }

    override fun computeBackwardResult(withinBatch: Int, chain: FloatArray, result: FloatArray) {

        Arrays.fill(result, 0f)

        for (indexRow in 0..this.numberInputRows - 1) {

            result[this.maxRowIndices[indexRow]] = chain[indexRow]

        }

    }

}