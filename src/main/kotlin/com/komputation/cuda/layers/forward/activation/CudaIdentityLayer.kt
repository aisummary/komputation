package com.komputation.cuda.layers.forward.activation

import jcuda.Pointer

class CudaIdentityLayer internal constructor(name : String? = null, numberRows : Int, numberColumns : Int) : BaseCudaActivationLayer(name) {

    override var deviceForwardResult = Pointer()
    override val numberInputRows = numberRows
    override val maximumInputColumns = numberColumns

    override var deviceBackwardResult = Pointer()
    override val numberOutputRows = numberRows
    override val maximumOutputColumns = numberColumns

    override fun forward(batchSize: Int, deviceInput: Pointer, isTraining: Boolean): Pointer {

        this.deviceForwardResult = deviceInput

        return this.deviceForwardResult

    }

    override fun backward(batchSize: Int, chain: Pointer) : Pointer {

        this.deviceBackwardResult = chain

        return this.deviceBackwardResult

    }

}