package com.komputation.cuda.layers.continuation.stack

import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.copyFloatArrayFromDeviceToDevice
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.KernelLaunchConfiguration
import com.komputation.cuda.kernels.launch.computeColumnwiseLaunchConfiguration
import com.komputation.cuda.kernels.launch.computeEntrywiseLaunchConfiguration
import com.komputation.cuda.layers.CudaContinuation
import com.komputation.cuda.layers.continuation.BaseCudaContinuation
import com.komputation.optimization.Optimizable
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class CudaStack(
    name : String?,
    numberInputRows : Int,
    numbersOutputRows : IntArray,
    maximumInputColumns : Int,
    maximumOutputColumns : Int,
    private val createCopyBlockKernel: () -> Kernel,
    private val createAddKernel: () -> Kernel,
    private val layers: Array<CudaContinuation>,
    private val numberMultiprocessors : Int,
    private val numberResidentWarps : Int,
    private val warpSize : Int,
    private val maximumNumberThreads : Int) : BaseCudaContinuation(
    name,
    numberInputRows,
    numbersOutputRows.sum(),
    maximumInputColumns,
    maximumOutputColumns), Optimizable {

    override var deviceForwardLengths = Pointer()

    private var copyBlockKernel : Kernel? = null
    private var addKernel : Kernel? = null

    private val numberLayers = this.layers.size
    private val firstLayer = this.layers.first()
    private val pointersToFirstRowInOutput: Array<Pointer>

    private val pointersToLayerMaximumOutputEntries: Array<Pointer>
    private val pointersToLayerNumberOutputRows: Array<Pointer>
    private val outputLaunchConfigurations: Array<KernelLaunchConfiguration>
    private val pointersToOutputNumberIterations: Array<Pointer>

    private val pointerToZero = Pointer.to(intArrayOf(0))

    private val deviceIndividualChains = Array(this.numberLayers) { Pointer() }
    private val pointersToIndividualChains = Array(this.numberLayers) { index -> Pointer.to(this.deviceIndividualChains[index]) }

    override val deviceForwardResult = Pointer()
    private val pointerToForwardResult = Pointer.to(this.deviceForwardResult)
    override val deviceBackwardResult = Pointer()
    private val pointerToBackwardResult = Pointer.to(this.deviceBackwardResult)

    private var addKernelLaunchConfiguration : KernelLaunchConfiguration? = null
    private val addNumberIterations = intArrayOf(1)
    private val pointerToNumberAddIterations = Pointer.to(this.addNumberIterations)

    init {
        val firstRowsInOutput = IntArray(this.numberLayers)

        numbersOutputRows.foldIndexed(0) { index, oldTotal, numberOutputRows ->
            firstRowsInOutput[index] = oldTotal
            oldTotal.plus(numberOutputRows)
        }

        this.pointersToFirstRowInOutput = Array(this.numberLayers) { index -> Pointer.to(intArrayOf(firstRowsInOutput[index])) }

        this.pointersToLayerMaximumOutputEntries = Array(this.numberLayers) { index -> Pointer.to(intArrayOf(this.layers[index].maximumOutputEntries)) }
        this.pointersToLayerNumberOutputRows = Array(this.numberLayers) { index -> Pointer.to(intArrayOf(this.layers[index].numberOutputRows)) }
        this.outputLaunchConfigurations = Array(this.numberLayers) { index ->
            computeColumnwiseLaunchConfiguration(numbersOutputRows[index], maximumOutputColumns, maximumNumberThreads)
        }
        this.pointersToOutputNumberIterations = Array(this.numberLayers) { index -> Pointer.to(intArrayOf(this.outputLaunchConfigurations[index].numberIterations)) }
    }

    override fun acquire(maximumBatchSize: Int) {
        super.acquire(maximumBatchSize)

        allocateDeviceFloatMemory(this.deviceForwardResult, this.forwardResultSize)
        allocateDeviceFloatMemory(this.deviceBackwardResult, this.backwardResultSize)

        this.copyBlockKernel = this.createCopyBlockKernel()
        this.addKernel = this.createAddKernel()

        this.deviceIndividualChains.forEachIndexed { index, pointer ->
            allocateDeviceFloatMemory(pointer, maximumBatchSize * this.layers[index].maximumOutputEntries)
        }

        this.addKernelLaunchConfiguration = computeEntrywiseLaunchConfiguration(this.backwardResultSize, this.numberMultiprocessors, this.numberResidentWarps, this.warpSize, this.maximumNumberThreads)
        this.addNumberIterations[0] = this.addKernelLaunchConfiguration!!.numberIterations
    }

    override fun release() {
        super.release()

        cudaFree(this.deviceForwardResult)
        cudaFree(this.deviceBackwardResult)

        this.deviceIndividualChains.forEach { chain ->
            cudaFree(chain)
        }

        this.copyBlockKernel!!.destroy()
        this.addKernel!!.destroy()
    }

    private val batchSize = IntArray(1)
    private val pointerToBatchSize = Pointer.to(this.batchSize)

    override fun forward(batchSize: Int, deviceInput: Pointer, deviceInputLengths: Pointer, isTraining: Boolean): Pointer {
        this.batchSize[0] = batchSize

        for ((index, layer) in (this.layers.withIndex())) {
            val individualForwardResult = layer.forward(batchSize, deviceInput, deviceInputLengths, isTraining)

            val launchConfiguration = this.outputLaunchConfigurations[index]

            // Copy the individual result to the stacked result (using the block that starts at the given first row)
            this.copyBlockKernel!!.launch(
                Pointer.to(
                    this.pointerToBatchSize,
                    this.pointersToOutputNumberIterations[index],
                    Pointer.to(individualForwardResult),
                    this.pointersToLayerMaximumOutputEntries[index],
                    this.pointersToLayerNumberOutputRows[index],
                    this.pointerToZero,
                    this.pointerToForwardResult,
                    this.pointerToMaximumOutputEntries,
                    this.pointerToNumberOutputRows,
                    this.pointersToFirstRowInOutput[index]
                ),
                this.maximumBatchSize,
                launchConfiguration.numberBlocks,
                launchConfiguration.numberThreadsPerBlock,
                0
            )
        }

        this.deviceForwardLengths = this.firstLayer.deviceForwardLengths

        return this.deviceForwardResult
    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {

        // Break up the chain
        (0 until this.numberLayers).forEach { index ->
            val launchConfiguration = this.outputLaunchConfigurations[index]

            this.copyBlockKernel!!.launch(
                Pointer.to(
                    this.pointerToBatchSize,
                    this.pointersToOutputNumberIterations[index],
                    Pointer.to(chain),
                    this.pointerToMaximumOutputEntries,
                    this.pointerToNumberOutputRows,
                    this.pointersToFirstRowInOutput[index],
                    this.pointersToIndividualChains[index],
                    this.pointersToLayerMaximumOutputEntries[index],
                    this.pointersToLayerNumberOutputRows[index],
                    this.pointerToZero),
                this.maximumBatchSize,
                launchConfiguration.numberBlocks,
                launchConfiguration.numberThreadsPerBlock,
                0)
        }

        // Copy the first individual backward result to the total backward result
        val firstBackwardResult = this.firstLayer.backward(batchSize, this.deviceIndividualChains.first())

        if (this.numberLayers == 1) {
            copyFloatArrayFromDeviceToDevice(firstBackwardResult, this.deviceBackwardResult, this.backwardResultSize)
        }
        else {
            // Add the remaining individual backward results to the total backward result
            var previousBackwardResult = firstBackwardResult
            for (index in 1 until this.numberLayers) {
                val currentBackwardResult = this.layers[index].backward(batchSize, this.deviceIndividualChains[index])

                this.addKernel!!.launch(
                    Pointer.to(
                        Pointer.to(previousBackwardResult),
                        Pointer.to(currentBackwardResult),
                        this.pointerToBackwardResult,
                        this.pointerToNumberAddIterations,
                        this.pointerToBackwardResultSize
                    ),
                    this.addKernelLaunchConfiguration!!.numberBlocks,
                    1,
                    this.addKernelLaunchConfiguration!!.numberThreadsPerBlock,
                    0)

                previousBackwardResult = currentBackwardResult

            }
        }

        return this.deviceBackwardResult
    }

    private val optimizableLayers = this.layers.filterIsInstance(Optimizable::class.java)

    override fun optimize(batchSize: Int) {
        this.optimizableLayers.forEach { layer -> layer.optimize(batchSize) }
    }


}