package com.komputation.cuda.layers.forward.projection

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import com.komputation.cuda.*
import com.komputation.cuda.functions.cublasBackwardProjectionWrtBias
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeNumberOfThreadsForRows
import com.komputation.cuda.layers.BaseCudaForwardLayer
import com.komputation.cuda.layers.CudaVariableLengthForwardLayer
import com.komputation.cuda.optimization.BaseCudaUpdateRule
import com.komputation.layers.Resourceful
import com.komputation.optimization.Optimizable

class CublasBiasLayer internal constructor(
    name: String?,
    private val cublasHandle: cublasHandle,
    numberRows: Int,
    numberColumns: Int,
    private val initialBias: FloatArray,
    private val biasUpdateRule: BaseCudaUpdateRule?,
    private val createKernel: () -> Kernel,
    private val warpSize : Int,
    private val maximumNumberThreadsPerBlock: Int) : BaseCudaForwardLayer(name), CudaVariableLengthForwardLayer, Optimizable, Resourceful {

    private val numberEntries = numberRows * numberColumns

    private var kernel : Kernel? = null

    private var numberBlocksInXDimension = -1
    private var numberBlocksInYDimension = -1
    private var numberThreadsPerBlock = -1

    override val deviceForwardResult = Pointer()
    override val numberOutputRows = numberRows
    override val maximumOutputColumns = numberColumns
    private val pointerToDeviceForwardResult = Pointer.to(this.deviceForwardResult)

    private val deviceBias = Pointer()
    private val pointerToDeviceBias = Pointer.to(this.deviceBias)

    override val deviceBackwardResult = Pointer()
    override val numberInputRows = numberRows
    override val maximumInputColumns = numberColumns
    private val pointerToDeviceBackwardWrtBias = Pointer.to(this.deviceBackwardResult)

    private val deviceOnes = Pointer()

    private val pointerToNumberEntries = Pointer.to(intArrayOf(this.numberEntries))
    private val pointerToNumberInputRows = Pointer.to(intArrayOf(this.numberInputRows))

    private val batchSize = intArrayOf(-1)
    private val pointerToBatchSize = Pointer.to(this.batchSize)

    private val numberIterations = intArrayOf(-1)
    private val pointerToNumberIterations = Pointer.to(this.numberIterations)

    private var numberBatchInputColumns = -1
    private var maximumBatchSize = -1
    private val deviceMaximumInputColumns = Pointer()
    private val pointerToMaximumInputColumns = Pointer.to(deviceMaximumInputColumns)

    override fun acquire(maximumBatchSize : Int) {

        this.maximumBatchSize = maximumBatchSize

        this.numberBatchInputColumns = maximumBatchSize * this.maximumInputColumns

        this.kernel = this.createKernel()

        setIntArray(IntArray(maximumBatchSize) { this.maximumInputColumns }, this.maximumBatchSize, this.deviceMaximumInputColumns)

        setFloatArray(this.initialBias, this.numberEntries, this.deviceBias)

        val numberBatchResultEntries = maximumBatchSize * this.numberEntries
        allocateDeviceFloatMemory(this.deviceForwardResult, numberBatchResultEntries)

        allocateDeviceFloatMemory(this.deviceBackwardResult, this.numberEntries)

        this.biasUpdateRule?.acquire(maximumBatchSize)

        setFloatArray(FloatArray(this.numberBatchInputColumns) { 1f }, this.numberBatchInputColumns, this.deviceOnes)

        this.numberBlocksInXDimension = maximumBatchSize
        this.numberBlocksInYDimension = this.maximumInputColumns

        val (numberIterations, numberThreadsPerBlock) = computeNumberOfThreadsForRows(this.numberInputRows, this.warpSize, this.maximumNumberThreadsPerBlock)

        this.numberThreadsPerBlock = numberIterations
        this.numberIterations[0] = numberThreadsPerBlock

    }

    override fun forward(batchSize: Int, deviceInput: Pointer, isTraining: Boolean): Pointer {

        this.batchSize[0] = batchSize

        this.kernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                this.pointerToMaximumInputColumns,
                this.pointerToNumberEntries,
                this.pointerToNumberInputRows,
                this.pointerToNumberIterations,
                Pointer.to(deviceInput),
                this.pointerToDeviceBias,
                this.pointerToDeviceForwardResult
            ),
            this.numberBlocksInXDimension,
            this.numberBlocksInYDimension,
            this.numberThreadsPerBlock,
            0
        )

        return this.deviceForwardResult

    }

    override fun forward(batchSize: Int, deviceLengths: Pointer, deviceInput: Pointer, isTraining: Boolean): Pointer {

        this.batchSize[0] = batchSize

        this.kernel!!.launch(
            Pointer.to(
                this.pointerToBatchSize,
                Pointer.to(deviceLengths),
                this.pointerToNumberEntries,
                this.pointerToNumberInputRows,
                this.pointerToNumberIterations,
                Pointer.to(deviceInput),
                this.pointerToDeviceBias,
                this.pointerToDeviceForwardResult
            ),
            this.numberBlocksInXDimension,
            this.numberBlocksInYDimension,
            this.numberThreadsPerBlock,
            0
        )

        return this.deviceForwardResult

    }


    override fun backward(batchSize: Int, chain: Pointer): Pointer {

        if(batchSize < this.maximumBatchSize) {

            setArrayToZero(this.deviceBackwardResult, this.numberEntries)

            cublasBackwardProjectionWrtBias(
                this.cublasHandle,
                chain,
                this.numberInputRows,
                batchSize * this.maximumInputColumns,
                this.deviceOnes,
                this.deviceBackwardResult)

        }
        else {

            cublasBackwardProjectionWrtBias(
                this.cublasHandle,
                chain,
                this.numberInputRows,
                this.numberBatchInputColumns,
                this.deviceOnes,
                this.deviceBackwardResult)

        }

        return this.deviceBackwardResult

    }

    override fun optimize(batchSize: Int) {

        this.biasUpdateRule?.denseUpdate(
            batchSize,
            this.pointerToDeviceBias,
            this.pointerToDeviceBackwardWrtBias)

    }

    override fun release() {

        cudaFree(this.deviceForwardResult)
        cudaFree(this.deviceBackwardResult)
        cudaFree(this.deviceBias)
        cudaFree(this.deviceOnes)

        this.kernel!!.destroy()

    }

}