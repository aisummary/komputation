package com.komputation.cpu.layers.forward.activation

import com.komputation.cpu.layers.BaseCpuForwardLayer

class CpuIdentityLayer internal constructor(name : String? = null, private val numberRows : Int, private val numberColumns : Int) : BaseCpuForwardLayer(name), CpuActivationLayer {

    override val numberOutputRows = this.numberRows
    override val numberOutputColumns = this.numberColumns
    override var forwardResult = FloatArray(0)

    override val numberInputRows = this.numberRows
    override val numberInputColumns = this.numberColumns
    override var backwardResult = FloatArray(0)

    override fun forward(withinBatch : Int, numberInputColumns : Int, input : FloatArray, isTraining : Boolean): FloatArray {

        this.forwardResult = input

        return this.forwardResult

    }

    override fun backward(withinBatch : Int, chain : FloatArray): FloatArray {

        this.backwardResult = chain

        return this.backwardResult

    }

}