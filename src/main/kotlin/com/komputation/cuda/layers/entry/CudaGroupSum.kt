package com.komputation.cuda.layers.entry

import jcuda.Pointer
import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.kernels.Kernel
import com.komputation.layers.Resourceful
import jcuda.runtime.JCuda.cudaFree

class CudaGroupSum(
    private val dimension: Int,
    private val maximumKeys: Int,
    private val hashTableSize : Int,
    private val createGroupSumKernel : () -> Kernel,
    private val createResetKernel : () -> Kernel) : Resourceful {

    private val pointerToNumberRows = Pointer.to(intArrayOf(this.dimension))
    private val pointerToMaximumKeys = Pointer.to(intArrayOf(this.maximumKeys))
    private val pointerToNumberGroupSumEntries = Pointer.to(intArrayOf(this.hashTableSize * this.dimension))

    private var groupSumKernel: Kernel? = null
    private var resetKernel: Kernel? = null

    private val deviceGroupSum = Pointer()
    fun getDeviceSum() = this.deviceGroupSum
    private val pointerToGroupSum = Pointer.to(this.deviceGroupSum)
    fun getPointerToSum() = this.pointerToGroupSum

    private val pointerToZero = Pointer.to(floatArrayOf(0f))

    private var maximumBatchSize = -1

    override fun acquire(maximumBatchSize: Int) {

        this.maximumBatchSize = maximumBatchSize

        allocateDeviceFloatMemory(this.deviceGroupSum, this.maximumBatchSize * this.hashTableSize * dimension)

        this.groupSumKernel = this.createGroupSumKernel()
        this.resetKernel = this.createResetKernel()

    }

    fun reset() {

        this.resetKernel!!.launch(
            Pointer.to(
                this.pointerToNumberRows,
                this.pointerToNumberGroupSumEntries,
                this.pointerToGroupSum,
                this.pointerToZero
            ),
            this.maximumBatchSize,
            this.hashTableSize,
            this.dimension,
            0)

    }

    fun sum(pointerToMapping : Pointer, pointerToInput: Pointer) {

        this.groupSumKernel!!.launch(
            Pointer.to(
                this.pointerToNumberRows,
                this.pointerToMaximumKeys,
                pointerToMapping,
                pointerToInput,
                this.pointerToGroupSum
            ),
            this.maximumBatchSize,
            this.maximumKeys,
            this.dimension,
            0)

    }

    override fun release() {

        this.maximumBatchSize = -1

        cudaFree(this.deviceGroupSum)

        this.groupSumKernel!!.destroy()
        this.resetKernel!!.destroy()

    }

}