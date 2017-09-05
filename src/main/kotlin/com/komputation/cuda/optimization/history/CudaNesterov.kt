
package com.komputation.cuda.optimization.history

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeEntrywiseLaunchConfiguration
import com.komputation.cuda.optimization.BaseCudaUpdateRule
import com.komputation.layers.Resourceful

class CudaNesterov internal constructor(
    private val numberParameters : Int,
    private val parameterSize : Int,
    private val learningRate: Float,
    private val momentum : Float,
    private val createKernel: () -> Kernel,
    private val numberMultiprocessors : Int,
    private val numberResidentWarps : Int,
    private val warpSize : Int,
    private val maximumNumberThreads : Int) : BaseCudaUpdateRule(), Resourceful {

    private val pointerToParameterSize = Pointer.to(intArrayOf(this.parameterSize))
    private val pointerToLearningRate = Pointer.to(floatArrayOf(this.learningRate))
    private val pointerToMomentum = Pointer.to(floatArrayOf(this.momentum))

    private val deviceHistory = Pointer()
    private val pointerToHistory = Pointer.to(this.deviceHistory)

    private val deviceBackup = Pointer()
    private val pointerToBackup = Pointer.to(this.deviceBackup)

    private var kernel : Kernel? = null
    private var numberBlocks = -1
    private var numberThreads = -1
    private val numberIterations = intArrayOf(-1)
    private var pointerToNumberIterations = Pointer.to(this.numberIterations)

    override fun acquire(maximumBatchSize : Int) {

        super.acquire(maximumBatchSize)

        this.kernel = this.createKernel()

        val launchConfiguration = computeEntrywiseLaunchConfiguration(this.parameterSize, this.numberMultiprocessors, this.numberResidentWarps, this.warpSize, this.maximumNumberThreads)
        this.numberBlocks = launchConfiguration.numberBlocks
        this.numberThreads = launchConfiguration.numberThreadsPerBlock
        this.numberIterations[0] = launchConfiguration.numberIterations

        val totalNumberEntries = this.numberParameters * this.parameterSize

        allocateDeviceFloatMemory(this.deviceHistory, totalNumberEntries)
        
        allocateDeviceFloatMemory(this.deviceBackup, totalNumberEntries)

    }

    override fun launchKernel(
        maximumParameters: Int,
        pointerToIndices : Pointer,
        pointerToCounts : Pointer,
        pointerToParameters: Pointer,
        pointerToGradient: Pointer) : Int {

        val parameters = Pointer.to(
            this.pointerToNumberIterations,
            this.pointerToLearningRate,
            this.pointerToMomentum,
            this.pointerToHistory,
            this.pointerToBackup,
            pointerToIndices,
            pointerToCounts,
            this.pointerToParameterSize,
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

        super.release()

        this.kernel!!.destroy()

        cudaFree(this.deviceHistory)
        cudaFree(this.deviceBackup)

    }

}