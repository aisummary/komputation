package com.komputation.cuda.layers.forward.maxpooling

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import com.komputation.cuda.*
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeRowwiseLaunchConfiguration
import com.komputation.cuda.layers.BaseCudaForwardLayer
import com.komputation.cuda.layers.CudaVariableLengthForwardLayer
import com.komputation.layers.Resourceful

class CudaMaxPoolingLayer internal constructor(
    name : String?,
    numberRows : Int,
    override val maximumInputColumns : Int,
    private val symbolForUnusedColumns : Float,
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    private val maximumNumberThreadsPerBlock: Int,
    private val warpSize : Int) : BaseCudaForwardLayer(name), Resourceful, CudaVariableLengthForwardLayer {

    private val maximumNumberEntries = numberRows * this.maximumInputColumns
    private val pointerToMaximumNumberEntries = Pointer.to(intArrayOf(this.maximumNumberEntries))

    private val pointerToNumberRows = Pointer.to(intArrayOf(numberRows))

    private var forwardKernel: Kernel? = null
    override val deviceForwardResult = Pointer()
    private val pointerToForwardResult = Pointer.to(this.deviceForwardResult)
    override val numberOutputRows = numberRows
    override val maximumOutputColumns = 1

    private var backwardKernel: Kernel? = null
    override val numberInputRows = numberRows

    private val batchSize = intArrayOf(-1)
    private val pointerToBatchSize = Pointer.to(this.batchSize)

    private val deviceMaxIndices = Pointer()
    private val pointerToMaxIndices = Pointer.to(this.deviceMaxIndices)

    private var forwardConfiguration = computeRowwiseLaunchConfiguration(this.numberInputRows, this.maximumInputColumns, this.maximumNumberThreadsPerBlock)
    private val numberWarps = (this.maximumInputColumns+this.warpSize-1)/this.warpSize
    private val forwardSharedMemoryBytes = computeDeviceIntArraySize(this.numberWarps).toInt()

    private var maximumBatchSize = -1

    override val deviceBackwardResult = Pointer()
    private val pointerToBackwardResult = Pointer.to(this.deviceBackwardResult)

    private val deviceMaximumBatchLengths = Pointer()
    private var pointerToMaximumBatchLengths = Pointer()

    private var pointerToForwardBatchLengths = Pointer()

    private val pointerToSymbolForUnusedColumns = Pointer.to(floatArrayOf(this.symbolForUnusedColumns))

    override fun acquire(maximumBatchSize : Int) {

        this.maximumBatchSize = maximumBatchSize
        this.forwardKernel = this.createForwardKernel()
        this.backwardKernel = this.createBackwardKernel()

        allocateDeviceIntMemory(this.deviceMaxIndices, maximumBatchSize * this.numberInputRows)
        allocateDeviceFloatMemory(this.deviceForwardResult, maximumBatchSize * this.numberInputRows)

        allocateDeviceFloatMemory(this.deviceBackwardResult, maximumBatchSize * this.maximumNumberEntries)

        val maximumBatchLengths = IntArray(maximumBatchSize) { this.maximumInputColumns }
        setIntArray(maximumBatchLengths, maximumBatchSize, this.deviceMaximumBatchLengths)
        this.pointerToMaximumBatchLengths = Pointer.to(this.deviceMaximumBatchLengths)

    }

    var count = 0

    override fun forward(batchSize: Int, deviceInput: Pointer, isTraining: Boolean): Pointer {

        this.batchSize[0] = batchSize

        this.forwardKernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                this.pointerToMaximumBatchLengths,
                this.pointerToMaximumNumberEntries,
                this.pointerToMaxIndices,
                Pointer.to(deviceInput),
                this.pointerToForwardResult
            ),
            this.maximumBatchSize,
            this.forwardConfiguration.numberBlocks,
            this.forwardConfiguration.numberThreadsPerBlock,
            this.forwardSharedMemoryBytes
        )

        this.pointerToForwardBatchLengths = this.pointerToMaximumBatchLengths

        return this.deviceForwardResult

    }

    override fun forward(batchSize: Int, deviceLengths: Pointer, deviceInput: Pointer, isTraining: Boolean): Pointer {

        this.batchSize[0] = batchSize

        val pointerToLengths = Pointer.to(deviceLengths)

        this.forwardKernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                pointerToLengths,
                this.pointerToMaximumNumberEntries,
                this.pointerToMaxIndices,
                Pointer.to(deviceInput),
                this.pointerToForwardResult
            ),
            this.maximumBatchSize,
            this.forwardConfiguration.numberBlocks,
            this.forwardConfiguration.numberThreadsPerBlock,
            this.forwardSharedMemoryBytes
        )

        this.pointerToForwardBatchLengths = pointerToLengths

        return this.deviceForwardResult

    }

    override fun backward(batchSize: Int, chain: Pointer) : Pointer {

        this.backwardKernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                this.pointerToForwardBatchLengths,
                this.pointerToSymbolForUnusedColumns,
                this.pointerToMaximumNumberEntries,
                this.pointerToNumberRows,
                this.pointerToMaxIndices,
                Pointer.to(chain),
                this.pointerToBackwardResult
            ),
            this.maximumBatchSize,
            this.numberInputRows,
            this.maximumInputColumns,
            0
        )

        return this.deviceBackwardResult

    }

    override fun release() {

        this.maximumBatchSize = -1
        this.forwardKernel!!.destroy()
        this.backwardKernel!!.destroy()

        cudaFree(this.deviceForwardResult)
        cudaFree(this.deviceBackwardResult)
        cudaFree(this.deviceMaxIndices)
        cudaFree(this.deviceMaximumBatchLengths)

    }

}