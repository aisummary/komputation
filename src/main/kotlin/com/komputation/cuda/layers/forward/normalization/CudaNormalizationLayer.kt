package com.komputation.cuda.layers.forward.normalization

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.computeDeviceFloatArraySize
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeColumnwiseLaunchConfiguration
import com.komputation.cuda.layers.forward.activation.BaseCudaActivationLayer
import com.komputation.layers.Resourceful

class CudaNormalizationLayer internal constructor(
    name : String? = null,
    numberRows : Int,
    numberColumns : Int,
    private val createForwardKernel: () -> Kernel,
    private val createBackwardKernel: (Int) -> Kernel,
    private val maximumNumberThreadsPerBlock: Int,
    private val warpSize: Int) : BaseCudaActivationLayer(name), Resourceful {

    private val numberEntries = numberRows * numberColumns

    private var forwardKernel : Kernel? = null
    override val numberOutputRows = numberRows
    override val maximumOutputColumns = numberColumns
    override val deviceForwardResult = Pointer()

    private val pointerToDeviceForwardResult = Pointer.to(this.deviceForwardResult)

    private var backwardKernel : Kernel? = null
    private val deviceSums = Pointer()
    private val pointerToDeviceSums = Pointer.to(this.deviceSums)

    override val deviceBackwardResult = Pointer()
    override val numberInputRows = numberRows
    override val maximumInputColumns = numberColumns
    private val pointerToDeviceBackwardResult = Pointer.to(this.deviceBackwardResult)

    private var numberBlocksInXDimensions = -1
    private var numberBlocksInYDimensions = -1
    private var numberThreads = -1
    private var forwardSharedMemoryBytes = -1
    private var backwardSharedMemoryBytes = -1

    private var numberIterations = intArrayOf(-1)
    private val pointerToNumberIterations = Pointer.to(this.numberIterations)

    override fun acquire(maximumBatchSize : Int) {

        val numberBatchEntries = maximumBatchSize * this.numberEntries
        val numberBatchColumns = maximumBatchSize * this.maximumInputColumns

        allocateDeviceFloatMemory(this.deviceForwardResult, numberBatchEntries)
        allocateDeviceFloatMemory(this.deviceSums, numberBatchColumns)

        allocateDeviceFloatMemory(this.deviceBackwardResult, numberBatchEntries)

        val launchConfiguration = computeColumnwiseLaunchConfiguration(this.numberInputRows, this.maximumInputColumns, this.maximumNumberThreadsPerBlock)

        this.numberBlocksInXDimensions = maximumBatchSize
        this.numberBlocksInYDimensions = launchConfiguration.numberBlocks
        this.numberThreads = launchConfiguration.numberThreadsPerBlock
        this.numberIterations[0] = launchConfiguration.numberIterations

        val numberWarps = (this.numberInputRows / launchConfiguration.numberIterations + this.warpSize - 1) / this.warpSize

        this.forwardSharedMemoryBytes = computeDeviceFloatArraySize(numberWarps).toInt()
        this.backwardSharedMemoryBytes = computeDeviceFloatArraySize(numberWarps).toInt()

        this.forwardKernel = this.createForwardKernel()
        this.backwardKernel = this.createBackwardKernel(this.numberThreads)

    }

    private val forwardBatchSize = intArrayOf(-1)
    private val pointerToForwardBatchSize = Pointer.to(this.forwardBatchSize)

    private val backwardBatchSize = intArrayOf(-1)
    private val pointerToBackwardBatchSize = Pointer.to(this.backwardBatchSize)

    private val pointerToNumberRows = Pointer.to(intArrayOf(this.numberInputRows))
    private val pointerToNumberEntries = Pointer.to(intArrayOf(this.numberEntries))

    override fun forward(batchSize: Int, deviceInput: Pointer, isTraining: Boolean): Pointer {

        this.forwardBatchSize[0] = batchSize

        val parameters = Pointer.to(
            this.pointerToForwardBatchSize,
            this.pointerToNumberRows,
            this.pointerToNumberEntries,
            this.pointerToNumberIterations,
            Pointer.to(deviceInput),
            this.pointerToDeviceSums,
            this.pointerToDeviceForwardResult
        )

        this.forwardKernel!!.launch(
            parameters,
            this.numberBlocksInXDimensions,
            this.numberBlocksInYDimensions,
            this.numberThreads,
            this.forwardSharedMemoryBytes)

        return this.deviceForwardResult

    }

    override fun backward(batchSize: Int, chain: Pointer) : Pointer {

        this.backwardBatchSize[0] = batchSize

        val parameters = Pointer.to(
            this.pointerToBackwardBatchSize,
            this.pointerToNumberRows,
            this.pointerToNumberEntries,
            this.pointerToNumberIterations,
            Pointer.to(chain),
            this.pointerToDeviceForwardResult,
            this.pointerToDeviceSums,
            this.pointerToDeviceBackwardResult
        )

        this.backwardKernel!!.launch(
            parameters,
            this.numberBlocksInXDimensions,
            this.numberBlocksInYDimensions,
            this.numberThreads,
            this.backwardSharedMemoryBytes)

        return this.deviceBackwardResult

    }

    override fun release() {

        cudaFree(this.deviceBackwardResult)

        this.backwardKernel!!.destroy()

        cudaFree(this.deviceForwardResult)
        cudaFree(this.deviceSums)

        this.forwardKernel!!.destroy()

    }

}