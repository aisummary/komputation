package com.komputation.cuda.layers.forward.dense

import jcuda.Pointer
import com.komputation.cuda.layers.BaseCudaForwardLayer
import com.komputation.cuda.layers.forward.activation.CudaActivationLayer
import com.komputation.cuda.layers.forward.projection.CublasProjectionLayer
import com.komputation.optimization.Optimizable

class CudaDenseLayer internal constructor(
    name: String?,
    private val projectionLayer: CublasProjectionLayer,
    private val activationLayer: CudaActivationLayer) : BaseCudaForwardLayer(name), Optimizable {

    override val deviceForwardResult
        get() = this.activationLayer.deviceForwardResult
    override val numberOutputRows
        get() = this.activationLayer.numberOutputRows
    override val maximumOutputColumns
        get() = this.activationLayer.maximumOutputColumns

    override val deviceBackwardResult
        get() = this.projectionLayer.deviceForwardResult
    override val numberInputRows
        get() = this.projectionLayer.numberInputRows
    override val maximumInputColumns
        get() = this.projectionLayer.maximumInputColumns

    override fun forward(batchSize: Int, deviceInput: Pointer, isTraining: Boolean): Pointer {

        val projected = this.projectionLayer.forward(batchSize, deviceInput, isTraining)

        val activated = this.activationLayer.forward(batchSize, projected, isTraining)

        return activated

    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {

        val backwardActivation = this.activationLayer.backward(batchSize, chain)

        val backwardProjection = this.projectionLayer.backward(batchSize, backwardActivation)

        return backwardProjection

    }

    override fun optimize(batchSize: Int) {

        this.projectionLayer.optimize(batchSize)

    }

}