package com.komputation.cuda.layers.forward.projection

import jcuda.Pointer
import com.komputation.cuda.layers.BaseCudaForwardLayer
import com.komputation.cuda.layers.CudaVariableLengthForwardLayer
import com.komputation.optimization.Optimizable

class CublasProjectionLayer internal constructor(
    name: String?,
    private val weightingLayer: CublasWeightingLayer,
    private val biasLayer: CublasBiasLayer) : BaseCudaForwardLayer(name), CudaVariableLengthForwardLayer, Optimizable {

    override val deviceForwardResult
        get() = this.biasLayer.deviceForwardResult
    override val numberOutputRows
        get() = this.biasLayer.numberOutputRows
    override val maximumOutputColumns
        get() = this.biasLayer.maximumOutputColumns

    override val deviceBackwardResult
        get() = this.weightingLayer.deviceBackwardResult
    override val numberInputRows
        get() = this.weightingLayer.numberInputRows
    override val maximumInputColumns
        get() = this.weightingLayer.maximumInputColumns

    override fun forward(batchSize: Int, deviceInput: Pointer, isTraining: Boolean): Pointer {

        val weighted = this.weightingLayer.forward(batchSize, deviceInput, isTraining)

        this.biasLayer.forward(batchSize, weighted, isTraining)

        return this.deviceForwardResult

    }

    override fun forward(batchSize: Int, deviceLengths: Pointer, deviceInput: Pointer, isTraining: Boolean): Pointer {

        val weighted = this.weightingLayer.forward(batchSize, deviceInput, isTraining)

        this.biasLayer.forward(batchSize, deviceLengths, weighted, isTraining)

        return this.deviceForwardResult

    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {

        this.biasLayer.backward(batchSize, chain)

        this.weightingLayer.backward(batchSize, chain)

        return this.deviceBackwardResult

    }

    override fun optimize(batchSize: Int) {

        this.weightingLayer.optimize(batchSize)

        this.biasLayer.optimize(batchSize)

    }

}