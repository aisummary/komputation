package com.komputation.cpu.layers.forward

import com.komputation.cpu.functions.transpose
import com.komputation.cpu.layers.BaseCpuForwardLayer
import com.komputation.layers.Resourceful

class CpuTranspositionLayer internal constructor(
    name : String? = null,
    private val numberRows : Int,
    private val numberColumns : Int) : BaseCpuForwardLayer(name), Resourceful {

    private val numberEntries = this.numberRows * this.numberColumns

    override val numberOutputRows = this.numberRows
    override val numberOutputColumns = this.numberColumns
    override var forwardResult = FloatArray(0)

    override val numberInputRows = this.numberColumns
    override val numberInputColumns = this.numberRows
    override var backwardResult = FloatArray(0)

    override fun acquire(maximumBatchSize: Int) {

        this.forwardResult = FloatArray(this.numberEntries)
        this.backwardResult = FloatArray(this.numberEntries)

    }

    override fun release() {


    }

    override fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining : Boolean): FloatArray {

        transpose(this.numberRows, this.numberColumns, input, this.forwardResult)

        return this.forwardResult

    }

    override fun backward(withinBatch : Int, chain: FloatArray): FloatArray {

        transpose(this.numberRows, this.numberColumns, chain, this.backwardResult)

        return this.backwardResult

    }

}