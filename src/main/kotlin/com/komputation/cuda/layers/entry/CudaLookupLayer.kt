package com.komputation.cuda.layers.entry

import jcuda.Pointer
import com.komputation.cpu.functions.concatenate
import com.komputation.cpu.functions.pad
import com.komputation.cuda.*
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeColumnwiseLaunchConfiguration
import com.komputation.cuda.layers.BaseCudaEntryPoint
import com.komputation.cuda.memory.InputMemory
import com.komputation.cuda.optimization.BaseCudaUpdateRule
import com.komputation.layers.Resourceful
import com.komputation.matrix.IntMatrix
import com.komputation.matrix.Matrix
import com.komputation.optimization.Optimizable
import jcuda.runtime.JCuda.cudaFree
import java.util.*

class CudaLookupLayer internal constructor(
    name : String?,
    private var vectors: FloatArray,
    maximumLength : Int,
    override val hasFixedLength: Boolean,
    dimension : Int,
    private val updateRule: BaseCudaUpdateRule?,
    private val createForwardKernel: () -> Kernel,
    private val hashing: CudaHashing,
    private val groupSum: CudaGroupSum,
    private val maximumNumberThreadsPerBlock : Int) : BaseCudaEntryPoint(name), CudaForwardState, Resourceful, Optimizable {

    private val numberVectorEntries = vectors.size

    override val deviceForwardResult = Pointer()
    private val pointerToForwardResult = Pointer.to(this.deviceForwardResult)

    override val numberOutputRows = dimension
    override val maximumOutputColumns = maximumLength

    private val numberEntries = this.numberOutputRows * this.maximumOutputColumns

    private var deviceIndices = Pointer()
    private var pointerToIndices = Pointer.to(this.deviceIndices)

    private var numbersOfColumns = IntArray(0)
    private var indices = IntArray(0)
    private var forwardResult = FloatArray(0)
    private var maximumBatchSize = intArrayOf(-1)
    private var forwardKernel: Kernel? = null

    private val deviceVectors = Pointer()
    private val pointerToDeviceVectors = Pointer.to(this.deviceVectors)

    private val launchConfiguration = computeColumnwiseLaunchConfiguration(
        this.numberOutputRows,
        this.maximumOutputColumns,
        this.maximumNumberThreadsPerBlock)

    private val pointerToMaximumBatchSize = Pointer.to(this.maximumBatchSize)
    private val pointerToMaximumLength = Pointer.to(intArrayOf(this.maximumOutputColumns))
    private val pointerToDimension = Pointer.to(intArrayOf(this.numberOutputRows))
    private val numberIterations = this.launchConfiguration.numberIterations
    private val pointerToNumberIterations = Pointer.to(intArrayOf(numberIterations))

    private var batchInputs = emptyArray<IntArray>()

    private var maximumParameters = -1

    override fun acquire(maximumBatchSize: Int) {

        this.maximumBatchSize[0] = maximumBatchSize

        this.numbersOfColumns = IntArray(maximumBatchSize)
        this.indices = IntArray(maximumBatchSize * this.maximumOutputColumns)

        this.forwardKernel = this.createForwardKernel()

        setFloatArray(this.vectors, this.numberVectorEntries, this.deviceVectors)

        val numberBatchEntries = maximumBatchSize * this.numberEntries
        allocateDeviceFloatMemory(this.deviceForwardResult, numberBatchEntries)

        this.batchInputs = Array(maximumBatchSize) { IntArray(0) }

        this.hashing.acquire(maximumBatchSize)
        this.maximumParameters = this.hashing.getMaximumKeys()

        this.groupSum.acquire(maximumBatchSize)

    }

    override fun release() {

        this.numbersOfColumns = IntArray(0)
        this.indices = IntArray(0)

        this.forwardResult = FloatArray(0)
        this.maximumBatchSize[0] = -1

        this.forwardKernel!!.destroy()

        this.vectors = getFloatArray(this.deviceVectors, this.numberVectorEntries)

        cudaFree(this.deviceVectors)

        this.hashing.release()
        this.maximumParameters = -1

        this.groupSum.release()

    }

    override fun forward(batchId : Int, batchSize : Int, batch: IntArray, inputs : Array<Matrix>, memory: InputMemory) : Pointer {

        val maximumBatchSize = this.maximumBatchSize[0]

        this.deviceIndices = getIndices(memory, batchId, batch, inputs, batchSize, maximumBatchSize)
        this.pointerToIndices = Pointer.to(this.deviceIndices)

        this.forwardKernel!!.launch(
            Pointer.to(
                this.pointerToDeviceVectors,
                this.pointerToIndices,
                this.pointerToForwardResult,
                this.pointerToMaximumBatchSize,
                this.pointerToMaximumLength,
                this.pointerToDimension,
                this.pointerToNumberIterations
            ),
            maximumBatchSize,
            this.launchConfiguration.numberBlocks,
            this.launchConfiguration.numberThreadsPerBlock,
            0)

        return this.deviceForwardResult

    }

    private fun getIndices(memory: InputMemory, batchId: Int, batch: IntArray, inputs: Array<Matrix>, batchSize: Int, maximumBatchSize: Int): Pointer {

        val optionalDeviceIndices = memory.tryToGetData(batchId)

        if(optionalDeviceIndices != null) {

            return optionalDeviceIndices

        }

        var batchNumberEntries = 0

        for ((withinBatch, id) in batch.withIndex()) {

            val input = inputs[id] as IntMatrix
            val inputEntries = input.entries

            val numberEntries = input.numberEntries

            this.batchInputs[withinBatch] = inputEntries
            this.numbersOfColumns[withinBatch] = numberEntries
            batchNumberEntries += numberEntries

            val finalEntries = if (this.hasFixedLength) {

                inputEntries

            }
            else {

                val paddedInputEntries = IntArray(this.maximumOutputColumns)
                pad(inputEntries, numberEntries, this.maximumOutputColumns, -1, paddedInputEntries)

                paddedInputEntries

            }

            concatenate(finalEntries, withinBatch * this.maximumOutputColumns, this.maximumOutputColumns, this.indices)

        }

        if (batchSize < this.maximumBatchSize[0]) {

            Arrays.fill(this.indices, batchSize * this.maximumOutputColumns, this.maximumBatchSize[0] * this.maximumOutputColumns, -1)

        }

        val deviceIndices = Pointer()
        setIntArray(this.indices, this.indices.size, deviceIndices)

        val deviceNumbersOfColumns = Pointer()
        setIntArray(this.numbersOfColumns, this.numbersOfColumns.size, deviceNumbersOfColumns)

        memory.setData(batchId, deviceIndices)

        if (!this.hasFixedLength) {

            val lengths = IntArray(maximumBatchSize) { index ->

                if (index < batchSize) {

                    inputs[batch[index]].numberEntries

                }
                else {

                    0

                }

            }

            val deviceLengths = Pointer()
            setIntArray(lengths, maximumBatchSize, deviceLengths)

            memory.setLengths(batchId, deviceLengths)

        }

        return deviceIndices

    }

    override fun backward(chain: Pointer) : Pointer {

        this.hashing.reset()
        this.hashing.hash(Pointer.to(this.deviceIndices))

        this.groupSum.reset()
        this.groupSum.sum(this.hashing.getPointerToMapping(), Pointer.to(chain))

        return chain

    }

    override fun optimize(batchSize: Int) {

        this.updateRule?.sparseUpdate(
            this.maximumParameters,
            this.hashing.getPointerToHashTable(),
            this.hashing.getPointerToCounts(),
            this.pointerToDeviceVectors,
            this.groupSum.getPointerToSum())

    }

}