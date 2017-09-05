package com.komputation.cuda

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import com.komputation.cuda.kernels.Kernel
import com.komputation.layers.Resourceful

class CudaEvaluation(
    private val numberInstances : Int,
    private val numberRows : Int,
    private val numberColumns : Int,
    private val createKernel: () -> Kernel) : Resourceful {

    private val deviceCorrectPredictions = Pointer()
    private var kernel : Kernel? = null

    private val pointerToNumberRows = Pointer.to(intArrayOf(this.numberRows))
    private val pointerToNumberColumns = Pointer.to(intArrayOf(this.numberColumns))
    private val numberEntriesPerInstance = this.numberColumns * this.numberRows
    private val pointerToNumberEntriesPerInstance = Pointer.to(intArrayOf(this.numberEntriesPerInstance))
    private val pointerToCorrectPredictions = Pointer.to(this.deviceCorrectPredictions)

    private var count = 0

    override fun acquire(maximumBatchSize: Int) {

        allocateDeviceIntMemory(this.deviceCorrectPredictions, this.numberInstances)

        this.kernel = this.createKernel()

    }

    fun evaluateBatch(batchSize: Int, pointerToPredictions : Pointer, pointerToTargets : Pointer) {

        val parameters = Pointer.to(
            Pointer.to(intArrayOf(this.count)),
            this.pointerToNumberRows,
            this.pointerToNumberColumns,
            this.pointerToNumberEntriesPerInstance,
            pointerToPredictions,
            pointerToTargets,
            this.pointerToCorrectPredictions
        )

        this.kernel!!.launch(
            parameters,
            batchSize,
            1,
            1,
            0)

        this.count += batchSize

    }

    fun resetCount() {

        this.count = 0

    }

    fun computeAccuracy() =

        getIntArray(this.deviceCorrectPredictions, this.numberInstances).sum().toFloat().div(this.numberInstances.toFloat())

    override fun release() {

        this.kernel!!.destroy()

        cudaFree(this.deviceCorrectPredictions)

    }
}