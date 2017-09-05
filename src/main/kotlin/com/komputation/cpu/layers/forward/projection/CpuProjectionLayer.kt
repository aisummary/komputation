package com.komputation.cpu.layers.forward.projection

import com.komputation.cpu.layers.BaseCpuForwardLayer
import com.komputation.optimization.Optimizable

class CpuProjectionLayer internal constructor(
    name : String? = null,
    private val weightingLayer: CpuWeightingLayer,
    private val biasLayer : CpuBiasLayer) : BaseCpuForwardLayer(name), Optimizable {

    override val numberOutputRows
        get() = this.biasLayer.numberOutputRows
    override val numberOutputColumns
        get() = this.biasLayer.numberOutputColumns
    override val forwardResult
        get() = this.biasLayer.forwardResult

    override val numberInputRows
        get() = this.weightingLayer.numberInputRows
    override val numberInputColumns
        get() = this.weightingLayer.numberInputColumns
    override val backwardResult
        get() = this.weightingLayer.backwardResult


    override fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining: Boolean): FloatArray {

        val weighted = this.weightingLayer.forward(withinBatch, numberInputColumns, input, isTraining)

        this.biasLayer.forward(withinBatch, numberInputColumns, weighted, isTraining)

        return this.forwardResult

    }

    override fun backward(withinBatch : Int, chain : FloatArray) : FloatArray {

        this.biasLayer.backward(withinBatch, chain)

        this.weightingLayer.backward(withinBatch, chain)

        return this.backwardResult

    }

    override fun optimize(batchSize : Int) {

        this.weightingLayer.optimize(batchSize)

        this.biasLayer.optimize(batchSize)

    }

}