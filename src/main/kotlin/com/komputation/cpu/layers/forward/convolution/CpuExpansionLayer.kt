package com.komputation.cpu.layers.forward.convolution

import com.komputation.cpu.functions.backwardExpansionForConvolution
import com.komputation.cpu.functions.computeNumberFilterColumnPositions
import com.komputation.cpu.functions.expandForConvolution
import com.komputation.cpu.layers.BaseCpuVariableLengthForwardLayer

class CpuExpansionLayer internal constructor(
    name : String? = null,
    numberInputRows : Int,
    minimumColumns : Int,
    maximumColumns : Int,
    private val numberFilterRowPositions: Int,
    filterLength: Int,
    private val filterWidth: Int,
    private val filterHeight: Int) : BaseCpuVariableLengthForwardLayer(name, numberInputRows, filterLength, minimumColumns, maximumColumns) {

    private var numberFilterColumnPositionsOverPossibleLengths = IntArray(0)

    override fun acquire(maximumBatchSize: Int) {

        this.numberFilterColumnPositionsOverPossibleLengths = IntArray(this.numberLengths) { index -> computeNumberFilterColumnPositions(this.lengths[index], this.filterWidth) }

        super.acquire(maximumBatchSize)

    }

    override fun computeNumberOutputColumns(lengthIndex: Int, length : Int) =

        this.numberFilterColumnPositionsOverPossibleLengths[lengthIndex] * this.numberFilterRowPositions


    /*
        Ex.:
        input:
        i_11 i_12 i_13
        i_21 i_22 i_23
        i_31 i_32 i_33

        expansion:
        i_11 i_12
        i_21 i_22
        i_31 i_32
        i_12 i_13
        i_22 i_23
        i_32 i_33
    */
    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean, result: FloatArray) {

        expandForConvolution(
            this.numberInputRows,
            input,
            this.filterWidth,
            this.filterHeight,
            this.numberFilterRowPositions,
            this.numberFilterColumnPositionsOverPossibleLengths[this.lengthIndex],
            result)

    }

    // d expansion / d input
    override fun computeBackwardResult(withinBatch: Int, chain: FloatArray, result: FloatArray) {

        backwardExpansionForConvolution(
            this.numberInputRows,
            result,
            this.filterHeight,
            this.numberFilterRowPositions,
            this.numberFilterColumnPositionsOverPossibleLengths[this.lengthIndex],
            chain,
            this.numberOutputRows)

    }

}