package com.komputation.cuda

import jcuda.driver.CUcontext
import jcuda.driver.CUdevice
import jcuda.driver.JCudaDriver.*
import com.komputation.cuda.kernels.KernelFactory
import com.komputation.cuda.kernels.KernelInstruction

data class CudaContext(
    val context: CUcontext,
    val computeCapabilities: Pair<Int, Int>,
    val numberMultiprocessors: Int,
    val maximumNumberOfResidentWarpsPerMultiprocessor: Int,
    val warpSize: Int,
    val maximumNumberOfBlocks: Int,
    val maximumNumberOfThreadsPerBlock: Int) {

    private val kernelFactory = KernelFactory(this.computeCapabilities)

    fun createKernel(instruction: KernelInstruction) =

        this.kernelFactory.create(instruction)

    fun destroy() {

        destroyCudaContext(this.context)

    }

}

fun setUpCudaContext(deviceId : Int = 0): CudaContext {

    initializeCuda()

    val device = getCudaDevice(deviceId)

    val context = createCudaContext(device)

    val computeCapability = queryComputeCapability(device)

    val numberOfMultiprocessor = queryNumberOfMultiprocessor(device)
    val warpSize = queryWarpSize(device)
    val maximumNumberOfBlocks = queryMaximumNumberOfBlocks(device)
    val maximumNumberOfResidentThreads = queryMaximumNumberOfResidentThreads(device)
    val maximumNumberOfResidentWarps = maximumNumberOfResidentThreads / warpSize
    val maximumNumberOfThreadsPerBlock = queryMaximumNumberOfThreadsPerBlock(device)

    return CudaContext(context, computeCapability, numberOfMultiprocessor, maximumNumberOfResidentWarps, warpSize, maximumNumberOfBlocks, maximumNumberOfThreadsPerBlock)

}

fun initializeCuda() {

    cuInit(0)

}

fun createCudaContext(device : CUdevice): CUcontext {

    val context = CUcontext()
    cuCtxCreate(context, 0, device)

    return context

}

fun destroyCudaContext(context : CUcontext) =

    cuCtxDestroy(context)