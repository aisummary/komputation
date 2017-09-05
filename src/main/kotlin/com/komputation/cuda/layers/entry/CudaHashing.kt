package com.komputation.cuda.layers.entry

import jcuda.Pointer
import com.komputation.cuda.setIntArray
import com.komputation.layers.Resourceful
import jcuda.runtime.JCuda.cudaFree
import com.komputation.cuda.computeDeviceIntArraySize
import com.komputation.cuda.kernels.Kernel

class CudaHashing(
    private val maximumColumns: Int,
    private val sizeMultiplier : Int,
    private val createHashingKernel: () -> Kernel,
    private val createFillTwoIntegerArraysKernel: () -> Kernel) : Resourceful {

    private val deviceMapping = Pointer()
    fun getDeviceMapping() = this.deviceMapping
    private val pointerToMapping = Pointer.to(this.deviceMapping)
    fun getPointerToMapping() = this.pointerToMapping

    private val deviceHashTable = Pointer()
    fun getDeviceHashTable() = this.deviceHashTable
    private val pointerToHashTable = Pointer.to(this.deviceHashTable)
    fun getPointerToHashTable() = this.pointerToHashTable

    private val deviceCounts = Pointer()
    fun getDeviceCounts() = this.deviceCounts
    private val pointerToCounts = Pointer.to(this.deviceCounts)
    fun getPointerToCounts() = this.pointerToCounts

    private var hashingKernel : Kernel? = null
    private var fillTwoIntegerArraysKernel : Kernel? = null
    private var maximumBatchSize = -1

    private val pointerToMinusOne = Pointer.to(intArrayOf(-1))
    private val pointerToZero = Pointer.to(intArrayOf(0))
    private val pointerToOne = Pointer.to(intArrayOf(1))

    private var numberEntriesPerInstance = intArrayOf(-1)
    private val pointerToNumberEntriesPerInstance = Pointer.to(this.numberEntriesPerInstance)

    private val hashTableSize = intArrayOf(-1)
    private val pointerToHashTableSize = Pointer.to(this.hashTableSize)

    fun getMaximumKeys() = this.hashTableSize[0]

    override fun acquire(maximumBatchSize: Int) {

        this.maximumBatchSize = maximumBatchSize

        this.numberEntriesPerInstance[0] = this.sizeMultiplier * this.maximumColumns

        val hashTableSize = this.maximumBatchSize * numberEntriesPerInstance[0]
        this.hashTableSize[0] = hashTableSize

        setIntArray(IntArray(hashTableSize) { -1 }, hashTableSize, this.deviceHashTable)
        setIntArray(IntArray(hashTableSize) { 0 }, hashTableSize, this.deviceCounts)

        val mappingSize = this.maximumBatchSize * this.maximumColumns
        setIntArray(IntArray(mappingSize) { -1 }, mappingSize, this.deviceMapping)

        this.hashingKernel = this.createHashingKernel()
        this.fillTwoIntegerArraysKernel = this.createFillTwoIntegerArraysKernel()

    }

    fun reset() {

        this.fillTwoIntegerArraysKernel!!.launch(
            Pointer.to(
                this.pointerToOne,
                this.pointerToNumberEntriesPerInstance,
                this.pointerToHashTable,
                this.pointerToMinusOne,
                this.pointerToCounts,
                this.pointerToZero
            ),
            this.maximumBatchSize,
            1,
            this.numberEntriesPerInstance[0],
            0)

    }

    fun hash(pointerToIndices : Pointer) {

        this.hashingKernel!!.launch(
            Pointer.to(
                pointerToIndices,
                this.pointerToHashTableSize,
                this.pointerToHashTable,
                this.pointerToCounts,
                this.pointerToMapping
            ),
            this.maximumBatchSize,
            1,
            this.maximumColumns,
            computeDeviceIntArraySize(this.hashTableSize[0]).toInt())

    }

    override fun release() {

        this.maximumBatchSize = -1

        cudaFree(this.deviceMapping)
        cudaFree(this.deviceHashTable)
        cudaFree(this.deviceCounts)

        this.hashingKernel!!.destroy()
        this.fillTwoIntegerArraysKernel!!.destroy()

    }

}