package com.komputation.cuda.loss

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.computeDeviceFloatArraySize
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeColumnwiseLaunchConfiguration
import com.komputation.cuda.kernels.launch.computeEntrywiseLaunchConfiguration

class CudaSquaredLoss internal constructor(
    private val numberRows : Int,
    private val numberColumns : Int,
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    private val numberMultiprocessors : Int,
    private val numberResidentWarps : Int,
    private val warpSize : Int,
    private val maximumNumberThreadsPerBlock : Int) : CudaLossFunction {

    private val numberEntries = this.numberRows * this.numberColumns
    private val pointerToNumberEntries = Pointer.to(intArrayOf(this.numberEntries))

    private val pointerToNumberRows = Pointer.to(intArrayOf(this.numberRows))

    private var maximumBatchSize = -1

    private var forwardKernel : Kernel? = null

    private val deviceForwardResult = Pointer()
    private val pointerToForwardResult = Pointer.to(this.deviceForwardResult)

    private val forwardBatchSize = intArrayOf(-1)
    private val pointerToForwardBatchSize = Pointer.to(this.forwardBatchSize)
    private var forwardNumberBlocks = -1
    private var forwardNumberThreadsPerBlock = -1
    private var forwardNumberIterations = intArrayOf(-1)
    private var pointerToForwardNumberIterations = Pointer.to(this.forwardNumberIterations)
    private var forwardSharedMemoryBytes = -1

    private var backwardKernel : Kernel? = null

    private val deviceBackwardResults = Pointer()
    private val pointerToBackwardResults = Pointer.to(this.deviceBackwardResults)

    private val backwardBatchSize = intArrayOf(-1)
    private val pointerToBackwardBatchSize = Pointer.to(this.backwardBatchSize)
    private var backwardNumberBlocks = -1
    private var backwardNumberIterations = intArrayOf(-1)
    private var pointerToBackwardNumberIterations = Pointer.to(this.backwardNumberIterations)
    private var backwardNumberThreadsPerBlock = -1

    override fun acquire(maximumBatchSize: Int) {

        this.maximumBatchSize = maximumBatchSize

        val forwardLaunchConfiguration = computeColumnwiseLaunchConfiguration(this.numberRows, this.numberColumns, this.maximumNumberThreadsPerBlock)
        this.forwardNumberBlocks = forwardLaunchConfiguration.numberBlocks
        this.forwardNumberThreadsPerBlock = forwardLaunchConfiguration.numberThreadsPerBlock
        this.forwardNumberIterations[0] = forwardLaunchConfiguration.numberIterations
        val numberForwardWarps = (this.numberRows / forwardLaunchConfiguration.numberIterations + this.warpSize - 1) / this.warpSize
        this.forwardSharedMemoryBytes =  computeDeviceFloatArraySize(numberForwardWarps).toInt()

        this.forwardKernel = this.createForwardKernel()

        allocateDeviceFloatMemory(this.deviceForwardResult, this.maximumBatchSize * this.numberColumns)

        val backwardLaunchConfiguration = computeEntrywiseLaunchConfiguration(this.numberEntries, this.numberMultiprocessors, this.numberResidentWarps, this.warpSize, this.maximumNumberThreadsPerBlock)
        this.backwardNumberBlocks = backwardLaunchConfiguration.numberBlocks
        this.backwardNumberThreadsPerBlock = backwardLaunchConfiguration.numberThreadsPerBlock
        this.backwardNumberIterations[0] = backwardLaunchConfiguration.numberIterations

        this.backwardKernel = this.createBackwardKernel()

        allocateDeviceFloatMemory(this.deviceBackwardResults, this.numberEntries * maximumBatchSize)

    }

    override fun release() {

        cudaFree(this.deviceBackwardResults)

        this.backwardKernel!!.destroy()

        cudaFree(this.deviceForwardResult)

        this.forwardKernel!!.destroy()

        this.maximumBatchSize = -1

    }

    // int batchSize, int numberRows, int numberEntriesPerInstance, int numberIterations
    override fun accumulate(pointerToPredictions: Pointer, pointerToTargets: Pointer, batchSize : Int) {

        this.forwardBatchSize[0] = batchSize

        val parameters = Pointer.to(
            this.pointerToForwardBatchSize,
            this.pointerToNumberRows,
            this.pointerToNumberEntries,
            this.pointerToForwardNumberIterations,
            pointerToPredictions,
            pointerToTargets,
            this.pointerToForwardResult)

        this.forwardKernel!!.launch(
            parameters,
            this.maximumBatchSize,
            this.forwardNumberBlocks,
            this.forwardNumberThreadsPerBlock,
            this.forwardSharedMemoryBytes)

    }

    override fun accessAccumulation(): Float {

        val sums = getFloatArray(this.deviceForwardResult, this.maximumBatchSize * this.numberColumns)

        var loss = 0.0f

        for(sum in sums) {

            loss += sum

        }

        return 0.5f * loss

    }

    override fun backward(pointerToPredictions: Pointer, pointerToTargets: Pointer, batchSize: Int): Pointer {

        this.backwardBatchSize[0] = batchSize

        val parameters = Pointer.to(
            this.pointerToBackwardBatchSize,
            this.pointerToNumberEntries,
            this.pointerToBackwardNumberIterations,
            pointerToPredictions,
            pointerToTargets,
            this.pointerToBackwardResults)

        this.backwardKernel!!.launch(
            parameters,
            this.maximumBatchSize,
            this.backwardNumberBlocks,
            this.backwardNumberThreadsPerBlock,
            0)

        return this.deviceBackwardResults

    }

}