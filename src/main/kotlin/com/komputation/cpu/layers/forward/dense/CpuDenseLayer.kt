package com.komputation.cpu.layers.forward.dense

import com.komputation.cpu.layers.BaseCpuForwardLayer
import com.komputation.cpu.layers.forward.activation.CpuActivationLayer
import com.komputation.cpu.layers.forward.projection.CpuProjectionLayer
import com.komputation.optimization.Optimizable

class CpuDenseLayer internal constructor(
    name : String?,
    private val projection : CpuProjectionLayer,
    private val activation: CpuActivationLayer) : BaseCpuForwardLayer(name), Optimizable {

    override val numberOutputRows
        get() = this.activation.numberOutputRows
    override val numberOutputColumns
        get() = this.activation.numberOutputColumns
    override val forwardResult
        get() = this.activation.forwardResult

    override val numberInputRows
        get() = this.projection.numberInputRows
    override val numberInputColumns
        get() = this.projection.numberInputColumns
    override val backwardResult
        get() = this.projection.backwardResult

    override fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining : Boolean): FloatArray {

        val projected = this.projection.forward(withinBatch, numberInputColumns, input, isTraining)

        return this.activation.forward(withinBatch, numberInputColumns, projected, isTraining)

    }

    override fun backward(withinBatch : Int, chain: FloatArray): FloatArray {

        this.activation.backward(withinBatch, chain)

        return this.projection.backward(withinBatch, this.activation.backwardResult)

    }

    override fun optimize(batchSize : Int) {

        this.projection.optimize(batchSize)

    }

}