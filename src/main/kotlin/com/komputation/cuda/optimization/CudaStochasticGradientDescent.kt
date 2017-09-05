package com.komputation.cuda.optimization

import jcuda.Pointer
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeEntrywiseLaunchConfiguration
import com.komputation.layers.Resourceful

class CudaStochasticGradientDescent internal constructor(
    private val size : Int,
    private val learningRate: Float,
    private val createKernel: () -> Kernel,
    private val numberMultiprocessors : Int,
    private val numberResidentWarps : Int,
    private val warpSize : Int,
    private val maximumNumberThreads : Int) : BaseCudaUpdateRule(), Resourceful {

    private val pointerToSize = Pointer.to(intArrayOf(this.size))
    private val pointerToLearningRate = Pointer.to(floatArrayOf(this.learningRate))

    private var kernel : Kernel? = null
    private var numberBlocks = -1
    private var numberThreads = -1
    private val numberIterations = intArrayOf(-1)
    private var pointerToNumberIterations = Pointer.to(this.numberIterations)

    override fun acquire(maximumBatchSize : Int) {

        super.acquire(maximumBatchSize)

        this.kernel = this.createKernel()

        val launchConfiguration = computeEntrywiseLaunchConfiguration(this.size, this.numberMultiprocessors, this.numberResidentWarps, this.warpSize, this.maximumNumberThreads)
        this.numberBlocks = launchConfiguration.numberBlocks
        this.numberThreads = launchConfiguration.numberThreadsPerBlock
        this.numberIterations[0] = launchConfiguration.numberIterations

    }

    override fun launchKernel(
        maximumParameters: Int,
        pointerToIndices: Pointer,
        pointerToCounts : Pointer,
        pointerToParameters: Pointer,
        pointerToGradient: Pointer) : Int {

        val parameters = Pointer.to(
            this.pointerToNumberIterations,
            this.pointerToLearningRate,
            pointerToIndices,
            pointerToCounts,
            this.pointerToSize,
            pointerToParameters,
            pointerToGradient
        )

        return this.kernel!!.launch(
            parameters,
            maximumParameters,
            this.numberBlocks,
            this.numberThreads,
            0
        )

    }

    override fun release() {

        this.kernel!!.destroy()

        super.release()

    }

}