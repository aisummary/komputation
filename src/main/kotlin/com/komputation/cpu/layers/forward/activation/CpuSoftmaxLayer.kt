package com.komputation.cpu.layers.forward.activation

import com.komputation.cpu.layers.BaseCpuForwardLayer
import com.komputation.cpu.layers.forward.normalization.CpuNormalizationLayer

class CpuSoftmaxLayer internal constructor(
    name : String? = null,
    private val exponentiationLayer: CpuExponentiationLayer,
    private val normalizationLayer: CpuNormalizationLayer) : BaseCpuForwardLayer(name), CpuActivationLayer {

    override val numberOutputRows
        get() = this.normalizationLayer.numberOutputRows
    override val numberOutputColumns
        get() = this.normalizationLayer.numberOutputColumns
    override val forwardResult
        get() = this.normalizationLayer.forwardResult

    override val numberInputRows
        get() = this.exponentiationLayer.numberInputRows
    override val numberInputColumns
        get() = this.exponentiationLayer.numberInputColumns
    override val backwardResult
        get() = this.exponentiationLayer.backwardResult

    override fun forward(withinBatch : Int, numberInputColumns : Int, input : FloatArray, isTraining : Boolean): FloatArray {

        this.exponentiationLayer.forward(withinBatch, numberInputColumns, input, isTraining)

        val normalized = this.normalizationLayer.forward(withinBatch, this.exponentiationLayer.numberOutputColumns, this.exponentiationLayer.forwardResult, isTraining)

        return normalized

    }

    override fun backward(withinBatch : Int, chain : FloatArray): FloatArray {

        val backwardNormalization = this.normalizationLayer.backward(withinBatch, chain)

        val backwardExponentiation = this.exponentiationLayer.backward(withinBatch, backwardNormalization)

        return backwardExponentiation

    }

}